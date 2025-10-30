'''
Features:
 - Contextual Bandit for fast per-packet decisions
 - DQN or PPO/A2C for sequential decision making
 - Resource monitoring (CPU/memory)
 - Detection F1 score calculation
 - Deep-capture logging to PCAP

Usage:
    python hybrid_sniffer.py --interface any --output capture.pcap --rl dqn
    python hybrid_sniffer.py --rl bandit
    python hybrid_sniffer.py --rl ppo
"""
'''
import numpy as np
import random
from scapy.all import sniff, wrpcap, Ether, IP, TCP, UDP, ICMP, ARP, DNS, DNSQR
from collections import defaultdict, deque,namedtuple
from datetime import datetime
import argparse
import socket
import time
import math
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from sympy.physics.units import action

packet_counts = defaultdict(int)
packet_log = []
total_bytes = 0

class FeatureWindow:
    def __init__(self,window_size=200):
        self.window_size=window_size
        self.pkts=deque(maxlen=window_size)
        self.sizes=deque(maxlen=window_size)
        self.timestamps=deque(maxlen=window_size)
        self.srcs=deque(maxlen=window_size)
        self.syn_flags=deque(maxlen=window_size)
    def push(self,packet):
        now=time.time()
        self.timestamps.append(now)
        proto=0
        size=len(packet) if hasattr(packet,'__len__')else 0
        src=None
        if IP in packet:
            proto=packet[IP].proto
            size=getattr(packet[IP],'len',size)
            src=packet[IP].src
            if TCP in packet:
                flags=packet[TCP].flags
                self.syn_flags.append(1 if flags&0x02 else 0)
            else:
                self.syn_flags.append(0)
        else:
            self.syn_flags.append(0)
        if src is None:
            src=getattr(packet,'src','N/A')
        self.pkts.append(proto)
        self.sizes.append(size)
        self.srcs.append(src)
    def get_state(self):
        window_len=max(1,len(self.pkts))
        counts=defaultdict(int)
        for p in self.pkts:
            counts[p]+=1
        tcp=counts.get(6,0)/window_len
        udp=counts.get(17,0)/window_len
        icmp=counts.get(1,0)/window_len
        avg_size=float(np.mean(self.sizes)) if self.sizes else 0.0
        syn_rate=float(np.sum(self.syn_flags))/window_len
        uniq=len(set(self.srcs))
        src_entropy=math.log(uniq+1)
        return np.array([tcp,udp,icmp,avg_size/1500.0,syn_rate,src_entropy],dtype=np.float32)
def is_suspicious(packet):
    try:
        if IP in packet:
            if getattr(packet[IP],'len',0)>1500:return True
            if TCP in packet:
                flags=packet[TCP].flags
                if(flags&0x02)and not flags&0x10: return True
            if UDP in packet and len(packet[UDP].payload)>512: return True
    except: return False
    return False
class ResourceMonitor:
    def __init__(self):
        self.cpu,self.mem=[],[]
    def log(self):
        self.cpu.append(psutil.cpu_percent())
        self.mem.append(psutil.virtual_memory().percent)

def compute_f1(y_true,y_pred):
    TP=sum((yt==1 and yp==1) for yp,yt in zip(y_pred,y_true))
    FP=sum((yt==0 and yp==1) for yp,yt in zip(y_pred,y_true))
    FN=sum((yt==1 and yp==0) for yp,yt in zip(y_pred,y_true))
    return 2*TP/(2*TP+FP+FN+1e-8) if TP+FP+FN>0 else 0

class LinUCB:
    def __init__(self,n_actions=3,feature_dim=6,alpha=0.8):
        self.n_actions=n_actions
        self.feature_dim=feature_dim
        self.alpha=alpha
        self.A=[np.eye(feature_dim) for _ in range(n_actions)]
        self.b=[np.zeros(feature_dim) for _ in range(n_actions)]
    def select_action(self,state):
        p=np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv=np.linalg.inv(self.A[a])
            theta=A_inv @ self.b[a]
            p[a]=theta.dot(state)+self.alpha* np.sqrt(state.dot(A_inv).dot(state))
        return int(np.argmax(p))
    def update(self,action,state,reward):
        x=state.reshape(-1,1)
        self.A[action]+=x@x.T
        self.b[action]+=reward*state

transition=namedtuple('Transition',('state','action','reward','next_state','done'))
class ReplayBuffer:
    def __init__(self,capacity=10000):
        self.buffer=deque(maxlen=capacity)
    def push(self,*args):self.buffer.append(Transition(*args))
    def sample(self,batch_size):
        batch=random.sample(self.buffer,min(batch_size,len(self.buffer)))
        return Transition(*zip(*batch))
    def __len__(self):return len(self.buffer)

class DQN(nn.Module):
    def __init__(self,input_dim,n_actions,hidden=64):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,n_actions)
        )
    def forward(self,x):return self.net(x)

class DQNAgent:
    def __init__(self,state_dim,n_cations,lr=1e-3,gamma=0.99,epsilon_start=1.0,epsilon_final=0.05,epsilon_decay=2000,buffer_capacity=20000,batch_size=64):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net=DQN(state_dim,n_cations).to(self.device)
        self.target_net=DQN(state_dim,n_cations).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=lr)
        self.gamma=gamma
        self.epsilon_start=epsilon_start
        self.epsilon_decay=epsilon_decay
        self.epsilon_final=epsilon_final
        self.step_done=0
        self.replay=ReplayBuffer(buffer_capacity)
        self.batch_size=batch_size
        self.update_target_every=1000
    def select_action(self,state,train_mode=True):
        eps=self.epsilon_final+(self.epsilon_start-self.epsilon_final)*math.exp(-1.0*self.step_done/self.epsilon_decay)
        self.step_done+=1
        if train_mode and random.random()<eps:
            return random.randrange(self.policy_net.net[-1].out_features)
        self.policy_net.eval()
        with torch.no_grad():
            s=torch.tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
            return int(self.policy_net(s).argmax().item())
    def push_transition(self,*args):self.replay.push(*args)
    def train_step(self):
        if len(self.replay)<32:return None
        batch=self.replay.sample(self.batch_size)
        state=torch.tensor(batch.action,dtype=torch.long,device=self.device).unsqueeze(1)
        reward=torch.tensor(batch.reward,dtype=torch.float32,device=self.device).unsqueeze(1)
        next_state=torch.tensor(np.array(batch.next_state),dtype=torch.float32,device=self.device)
        done=torch.tensor(batch.done,dtype=torch.float32,device=self.device).unsqueeze(1)
        q_values=self.policy_net(state).gather(1,action)
        with torch.no_grad():
            next_q=self.target_net(next_state).max(1)[0].unsqueeze(1)
            q_target=reward+(1-done)*self.gamma*next_q
        loss=nn.functional.mse_loss(q_values,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.step_done%self.update_target_every==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()
    def save(self,path):
        torch.save({'policy':self.policy_net.state_dict(),'target':self.target_net.state_dict(),'optimizer':self.optimizer.state_dict(),'steps':self.step_done},path)
    def load(self,path):
        data=torch.load(path,map_location=self.device)
        self.policy_net.load_state_dict(data['policy'])
        self.target_net.load_state_dict(data['target'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.step_done=data.get('steps',0)

def hybrid_sniffer(interface='any',count=0,bpf_filter='',output='capture.pcap',rl_strategy='dqn'):
    fw=FeatureWindow()
    deep_captured=[]
    stats=defaultdict(int)
    total=0
    res_monitor=ResourceMonitor()
    y_true,y_pred=[],[]
    if rl_strategy=='bandit':agent=LinUCB(n_actions=3)
    else:agent=DQNAgent(state_dim=6,n_cations=3)

    def analyze_packet(packet):
        nonlocal total
        total+=1
        fw.push(packet)
        state=fw.get_state()
        label=is_suspicious(packet)
        y_true.append(1 if label else 0)
        if rl_strategy=='bandit':
            action=agent.select_action(state)
        else:
            action=agent.select_action(state,train_mode=True)
        reward=0.0
        if action==0:
            reward=-1.0 if label else -0.5
        elif action==1:
            stats['alerts']+=1
            reward=1.0 if label else -0.5
        elif action==2:
            deep_captured.append(packet)
            stats['deep_captures']+=1
            reward=1.0 if label else -0.2
        y_pred.append(1 if action>0 else 0)
        if rl_strategy !='bandit':
            next_state=fw.get_state()
            agent.push_transition(state,action,reward,next_state,False)
            if not total%8:
                loss=agent.train_step()
                if loss is not None and not total%200:
                    print(f"[TRAIN] step={agent.steps_done} replay_len={len(agent.replay)} loss={loss:.5f}")
        if not total%50==0: res_monitor.log()

    print(f"[+] Starting Hybrid RL Sniffer on interface={interface} RL={rl_strategy}")
    try:
        sniff(iface=None if interface == 'any' else interface,
              filter=bpf_filter if bpf_filter else None,
              prn=analyze_packet,
              store=False,
              count=count)
    except KeyboardInterrupt:
        print("\n[!] Stopped by user")
    finally:
        if deep_captured: wrpcap(output, deep_captured)
        if rl_strategy != 'bandit': agent.save(f"{rl_strategy}_model.pt")
        print("\n[Summary]")
        print(f"Total packets: {total}")
        print(f"Alerts: {stats.get('alerts', 0)}")
        print(f"Deep captures: {stats.get('deep_captures', 0)}")
        print(f"F1 score: {compute_f1(y_true, y_pred):.4f}")
        print(f"Avg CPU: {np.mean(res_monitor.cpu):.1f}%  Avg Mem: {np.mean(res_monitor.mem):.1f}%")
        print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Hybrid RL Packet Sniffer")
    parser.add_argument("-i", "--interface", type=str, default="any")
    parser.add_argument("-c", "--count", type=int, default=0)
    parser.add_argument("-f", "--filter", type=str, default="")
    parser.add_argument("-o", "--output", type=str, default="capture.pcap")
    parser.add_argument("--rl", type=str, choices=['dqn', 'bandit'], default='dqn', help="RL strategy")
    args = parser.parse_args()
    hybrid_sniffer(interface=args.interface, count=args.count, bpf_filter=args.filter, output=args.output,
                   rl_strategy=args.rl)


if __name__ == "__main__":
    main()