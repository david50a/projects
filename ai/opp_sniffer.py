#!/usr/bin/env python3
"""
PPO/A2C Packet Sniffer - Gym integration

Usage examples:

# Train PPO on a PCAP file (offline training):
python ppo_sniffer.py train-ppo --pcap dataset.pcap --timesteps 200000 --model ppo_sniffer.zip

# Train A2C on a PCAP:
python ppo_sniffer.py train-a2c --pcap dataset.pcap --timesteps 100000 --model a2c_sniffer.zip

# Evaluate a trained model on a PCAP and print F1/resource:
python ppo_sniffer.py eval-pcap --pcap dataset.pcap --model ppo_sniffer.zip --output eval_capture.pcap

# Run live evaluation (use --interface on Windows/Linux):
python ppo_sniffer.py eval-live --model ppo_sniffer.zip --interface any --output live_capture.pcap

# Quick bandit run on PCAP (no SB3):
python ppo_sniffer.py bandit --pcap dataset.pcap --output bandit_capture.pcap
"""
import argparse
import math
import time
import random
from collections import deque, defaultdict
from datetime import datetime

import numpy as np
import psutil
from scapy.all import sniff, wrpcap, rdpcap, IP, TCP, UDP, ICMP

# stable-baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces

# -----------------------------
# Feature window (same features as before)
# -----------------------------
class FeatureWindow:
    def __init__(self, window_size=200):
        self.window_size = window_size
        self.pkts = deque(maxlen=window_size)
        self.sizes = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.srcs = deque(maxlen=window_size)
        self.syn_flags = deque(maxlen=window_size)

    def push(self, packet):
        now = time.time()
        self.timestamps.append(now)
        proto = 0
        size = len(packet) if hasattr(packet, "__len__") else 0
        src = None
        try:
            if IP in packet:
                proto = packet[IP].proto
                size = getattr(packet[IP], "len", size)
                src = packet[IP].src
                if TCP in packet:
                    flags = packet[TCP].flags
                    self.syn_flags.append(1 if (flags & 0x02) else 0)
                else:
                    self.syn_flags.append(0)
            else:
                self.syn_flags.append(0)
        except Exception:
            self.syn_flags.append(0)

        if src is None:
            src = getattr(packet, "src", "N/A")

        self.pkts.append(proto)
        self.sizes.append(size)
        self.srcs.append(src)

    def reset(self):
        self.pkts.clear(); self.sizes.clear(); self.timestamps.clear(); self.srcs.clear(); self.syn_flags.clear()

    def get_state(self):
        window_len = max(1, len(self.pkts))
        counts = defaultdict(int)
        for p in self.pkts:
            counts[p] += 1
        tcp = counts.get(6, 0) / window_len
        udp = counts.get(17, 0) / window_len
        icmp = counts.get(1, 0) / window_len
        avg_size = float(np.mean(self.sizes)) if self.sizes else 0.0
        syn_rate = float(np.sum(self.syn_flags)) / window_len
        uniq = len(set(self.srcs))
        src_entropy = math.log(uniq + 1)
        # normalized/scale where needed
        return np.array([tcp, udp, icmp, avg_size / 1500.0, syn_rate, src_entropy], dtype=np.float32)

# -----------------------------
# Suspicious heuristic for labeling (used if dataset has no ground truth)
# -----------------------------
def is_suspicious(packet):
    try:
        if IP in packet:
            if getattr(packet[IP], "len", 0) > 1500:
                return True
            if TCP in packet:
                flags = packet[TCP].flags
                if (flags & 0x02) and not (flags & 0x10):
                    return True
            if UDP in packet and len(packet[UDP].payload) > 512:
                return True
    except Exception:
        return False
    return False

# -----------------------------
# Gym environment that draws packets from a PCAP (or iterable)
# -----------------------------
class SnifferEnv(gym.Env):
    """
    Gym environment that steps through packets from a packet source (list or generator).
    Observation: 6-dim state from FeatureWindow
    Action: Discrete(3) -> 0=ignore, 1=alert, 2=deep-capture
    Reward: shaped from heuristic (or external labels if provided)
    """
    metadata = {"render.modes": []}

    def __init__(self, packet_source, window_size=200, episode_length=500, use_labels=False, label_fn=None):
        super().__init__()
        self.feature_window = FeatureWindow(window_size=window_size)
        self.packet_source = packet_source  # iterator
        self.episode_length = episode_length
        self.steps = 0
        self.use_labels = use_labels
        self.label_fn = label_fn  # function(pkt) -> 0/1 if available

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # internal caches
        self._current_packet = None
        self.done = False

    def reset(self):
        self.steps = 0
        self.feature_window.reset()
        self.done = False
        self._current_packet = None
        # Seed the window with a few packets if possible
        for _ in range(min(10, self.episode_length)):
            try:
                pkt = next(self.packet_source)
                self.feature_window.push(pkt)
            except StopIteration:
                break
        return self.feature_window.get_state()

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called on finished environment")

        # consume next packet
        try:
            pkt = next(self.packet_source)
            self._current_packet = pkt
        except StopIteration:
            # no more packets: mark done
            self.done = True
            return self.feature_window.get_state(), 0.0, True, {}

        # update features with this packet
        self.feature_window.push(pkt)
        state = self.feature_window.get_state()

        # label either by provided label_fn or heuristic
        if self.use_labels and self.label_fn is not None:
            label = 1 if self.label_fn(pkt) else 0
        else:
            label = 1 if is_suspicious(pkt) else 0

        # reward shaping (customize as needed)
        if action == 0:  # ignore
            reward = -1.0 * label  # penalize missing a suspicious packet
        elif action == 1:  # alert
            reward = 1.0 * label - 0.3 * (1 - label)
        elif action == 2:  # deep-capture
            reward = 0.6 * label - 0.1 * (1 - label)
        else:
            reward = 0.0

        self.steps += 1
        done = self.steps >= self.episode_length
        self.done = done
        info = {"label": label}
        return state, float(reward), done, info

# -----------------------------
# Simple LinUCB contextual bandit (for completeness)
# -----------------------------
class LinUCB:
    def __init__(self, n_actions=3, feature_dim=6, alpha=0.8):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.A = [np.eye(feature_dim) for _ in range(n_actions)]
        self.b = [np.zeros(feature_dim) for _ in range(n_actions)]

    def select_action(self, state):
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta.dot(state) + self.alpha * np.sqrt(state.dot(A_inv).dot(state))
        return int(np.argmax(p))

    def update(self, action, state, reward):
        x = state.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * state

# -----------------------------
# Utilities: F1, resource monitor
# -----------------------------
def compute_f1(y_true, y_pred):
    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    denom = (2 * TP + FP + FN)
    return (2 * TP / denom) if denom > 0 else 0.0

class ResourceMonitor:
    def __init__(self):
        self.cpu = []
        self.mem = []

    def log(self):
        self.cpu.append(psutil.cpu_percent(interval=None))
        self.mem.append(psutil.virtual_memory().percent)

    def summary(self):
        return (np.mean(self.cpu) if self.cpu else 0.0, np.mean(self.mem) if self.mem else 0.0)

# -----------------------------
# Helpers: packet iterator factory
# -----------------------------
def pcap_packet_generator(pcap_path):
    packets = rdpcap(pcap_path)
    for pkt in packets:
        yield pkt
    # generator ends naturally

# -----------------------------
# Offline training function
# -----------------------------
def train_on_pcap(pcap_path, algorithm="ppo", timesteps=100_000, model_path="policy_model.zip", episode_length=500):
    print(f"[+] Loading pcap: {pcap_path}")
    pkt_gen = pcap_packet_generator(pcap_path)
    # make a fresh generator for envs: stable-baselines wraps env creation, so we create a function that creates a generator copy
    def make_env():
        # For stable training we want a generator that yields repeatedly; simplest approach is to read packets into a list
        packets = list(rdpcap(pcap_path))
        def gen():
            for p in packets:
                yield p
        return SnifferEnv(packet_source=gen(), window_size=200, episode_length=episode_length, use_labels=False)

    vec_env = DummyVecEnv([make_env])
    print(f"[+] Created env; training {algorithm.upper()} for {timesteps} timesteps...")

    if algorithm == "ppo":
        model = PPO("MlpPolicy", vec_env, verbose=1)
    else:
        model = A2C("MlpPolicy", vec_env, verbose=1)

    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    print(f"[+] Saved model to {model_path}")
    return model_path

# -----------------------------
# Evaluate model on PCAP (generate F1 and resource metrics)
# -----------------------------
def eval_on_pcap(pcap_path, model_path, use_sb3=True, output_pcap=None):
    print(f"[+] Evaluating model {model_path} on {pcap_path}")
    packets = list(rdpcap(pcap_path))
    pkt_iter = iter(packets)
    env = SnifferEnv(packet_source=pkt_iter, window_size=200, episode_length=len(packets), use_labels=False)
    fw = env.feature_window
    fw.reset()

    # load model
    model = None
    if use_sb3:
        # automatic detection: PPO or A2C
        model = PPO.load(model_path) if model_path.endswith(".zip") or model_path.endswith(".pt") else PPO.load(model_path)
    else:
        raise ValueError("Only stable-baselines3 models supported in eval_on_pcap")

    y_true = []
    y_pred = []
    deep_captured = []
    res = ResourceMonitor()
    count = 0

    # step through packets using env.step via its iterator logic
    # we will manually step: reset already seeded window inside env
    # we can't directly call env.step without feeding the internal generator; so call reset() then loop until StopIteration
    env.reset()  # seeds window (reads some packets)
    while True:
        try:
            # fetch next packet from packet_source via env.step cycle: we call model.predict on current state and then call env.step(action)
            state = fw.get_state()
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, done, info = env.step(int(action))
            # record labels/preds
            pkt = env._current_packet
            label = 1 if is_suspicious(pkt) else 0
            y_true.append(label)
            y_pred.append(1 if int(action) > 0 else 0)
            if int(action) == 2:
                deep_captured.append(pkt)
            count += 1
            if count % 50 == 0:
                res.log()
            if done:
                break
        except StopIteration:
            break

    f1 = compute_f1(y_true, y_pred)
    cpu_avg, mem_avg = res.summary()
    if output_pcap and deep_captured:
        wrpcap(output_pcap, deep_captured)
        print(f"[+] Deep-captured {len(deep_captured)} packets to {output_pcap}")

    print(f"[Eval Summary] Packets: {count} F1: {f1:.4f} Avg CPU: {cpu_avg:.1f}% Avg Mem: {mem_avg:.1f}%")
    return {"packets": count, "f1": f1, "cpu": cpu_avg, "mem": mem_avg}

# -----------------------------
# Live evaluation: load model and use it in analyze_packet
# -----------------------------
def run_live_eval(model_path, interface="any", output_pcap="live_capture.pcap", rl_algorithm="ppo"):
    print(f"[+] Running live evaluation with model {model_path} on interface={interface}")
    # load model
    model = None
    model = PPO.load(model_path) if rl_algorithm == "ppo" else A2C.load(model_path)
    fw = FeatureWindow(window_size=200)
    deep_captured = []
    stats = defaultdict(int)
    y_true = []
    y_pred = []
    res = ResourceMonitor()
    total = 0

    def analyze(packet):
        nonlocal total
        total += 1
        fw.push(packet)
        state = fw.get_state()
        action, _ = model.predict(state, deterministic=True)
        label = 1 if is_suspicious(packet) else 0
        y_true.append(label)
        y_pred.append(1 if int(action) > 0 else 0)

        if int(action) == 1:
            stats["alerts"] += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [ALERT] {packet.summary()}")
        elif int(action) == 2:
            stats["deep"] += 1
            deep_captured.append(packet)

        if total % 50 == 0:
            res.log()

    try:
        sniff(iface=None if interface == "any" else interface, prn=analyze, store=False)
    except KeyboardInterrupt:
        print("\n[!] Stopped by user")
    finally:
        if deep_captured:
            wrpcap(output_pcap, deep_captured)
            print(f"[+] Deep-captured saved to {output_pcap}")
        f1 = compute_f1(y_true, y_pred)
        cpu_avg, mem_avg = res.summary()
        print(f"[Summary] Packets: {total} Alerts: {stats.get('alerts',0)} Deeps: {stats.get('deep',0)} F1: {f1:.4f} CPU: {cpu_avg:.1f}% MEM: {mem_avg:.1f}%")

# -----------------------------
# Quick bandit-run on pcap (evaluation / simple training)
# -----------------------------
def run_bandit_on_pcap(pcap_path, output_pcap="bandit_capture.pcap"):
    pkt_list = list(rdpcap(pcap_path))
    gen = iter(pkt_list)
    fw = FeatureWindow(window_size=200)
    deep_captured = []
    bandit = LinUCB(n_actions=3)
    y_true, y_pred = [], []
    res = ResourceMonitor()
    count = 0

    # seed window a bit
    for _ in range(min(10, len(pkt_list))):
        try:
            fw.push(next(gen))
        except StopIteration:
            break

    # iterate remaining
    for pkt in gen:
        fw.push(pkt)
        state = fw.get_state()
        label = 1 if is_suspicious(pkt) else 0
        action = bandit.select_action(state)
        reward = 0.0
        if action == 0:
            reward = -1.0 if label else 0.0
        elif action == 1:
            reward = 1.0 if label else -0.5
        elif action == 2:
            deep_captured.append(pkt)
            reward = 1.0 if label else -0.2
        bandit.update(action, state, reward)
        y_true.append(label)
        y_pred.append(1 if action > 0 else 0)
        count += 1
        if count % 50 == 0:
            res.log()

    if deep_captured:
        wrpcap(output_pcap, deep_captured)
    f1 = compute_f1(y_true, y_pred)
    cpu_avg, mem_avg = res.summary()
    print(f"[Bandit Eval] Packets: {count} F1: {f1:.4f} CPU: {cpu_avg:.1f}% MEM: {mem_avg:.1f}% Deep-captured: {len(deep_captured)}")
    return {"packets": count, "f1": f1, "cpu": cpu_avg, "mem": mem_avg}

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="PPO/A2C Packet Sniffer with Gym")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train ppo
    p_ppo = sub.add_parser("train-ppo")
    p_ppo.add_argument("--pcap", required=True)
    p_ppo.add_argument("--timesteps", type=int, default=200_000)
    p_ppo.add_argument("--model", type=str, default="ppo_sniffer.zip")

    # train a2c
    p_a2c = sub.add_parser("train-a2c")
    p_a2c.add_argument("--pcap", required=True)
    p_a2c.add_argument("--timesteps", type=int, default=100_000)
    p_a2c.add_argument("--model", type=str, default="a2c_sniffer.zip")

    # eval pcap
    p_eval = sub.add_parser("eval-pcap")
    p_eval.add_argument("--pcap", required=True)
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--output", default=None)

    # live eval
    p_live = sub.add_parser("eval-live")
    p_live.add_argument("--model", required=True)
    p_live.add_argument("--interface", default="any")
    p_live.add_argument("--output", default="live_capture.pcap")
    p_live.add_argument("--algo", choices=["ppo","a2c"], default="ppo")

    # bandit
    p_band = sub.add_parser("bandit")
    p_band.add_argument("--pcap", required=True)
    p_band.add_argument("--output", default="bandit_capture.pcap")

    args = parser.parse_args()

    if args.cmd == "train-ppo":
        train_on_pcap(args.pcap, algorithm="ppo", timesteps=args.timesteps, model_path=args.model)
    elif args.cmd == "train-a2c":
        train_on_pcap(args.pcap, algorithm="a2c", timesteps=args.timesteps, model_path=args.model)
    elif args.cmd == "eval-pcap":
        eval_on_pcap(args.pcap, args.model, use_sb3=True, output_pcap=args.output)
    elif args.cmd == "eval-live":
        run_live_eval(args.model, interface=args.interface, output_pcap=args.output, rl_algorithm=args.algo)
    elif args.cmd == "bandit":
        run_bandit_on_pcap(args.pcap, output_pcap=args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
