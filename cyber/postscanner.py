from socket import *
import sys
from queue import Queue
import threading
import time
Q = Queue()
open_ports = []
''''''''''
def portscan(port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('192.168.1.171', port))
        return True
    except:
        return False


def fill_queue(port_list):
    for i in port_list:
        Q.put(i)


def work():
    while not Q.empty():
        port = Q.get()
        if portscan(port):
            print('Port {} is open!'.format(port))
            open_ports.append(port)
port_list=range(1,1024)
fill_queue(port_list)
thread_list = []
for t in range(1000):
    thread = threading.Thread(target=work)
    thread_list.append(thread)
for thread in thread_list:
    thread.start()
for thread in thread_list:
    thread.join()
print(open_ports)

for i in range(1,1024):
    result=portscan(i)
    if result:
        print('port {} is open!'.format(i))
    else:
        print('port {} is closed!'.format(i))
'''''''''
def portscanner():
    start=time.time()
    #target=input('Enter a host of scanning: ')
    t_ip=gethostbyname(gethostname())
    print('startig scanning on host: ',t_ip )
    for i in range(1,1001):
        s=socket(AF_INET,SOCK_STREAM)
        connection=s.connect_ex((t_ip,i))
        if(connection==0):
            print('port {} is open'.format(i))
        s.close()
    print('time taken: ',time.time()-start)
portscanner()
