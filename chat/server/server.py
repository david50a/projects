import socket
import threading
import sqlite3
import encryption
import traceback
import sha_256
lock=threading.Lock()
stop=threading.Event()
colors = [
    "red", "yellow", "magenta",
    "gray", "lightgray", "darkgray", "orange", "pink", "brown", "purple",
    "lightblue", "darkblue", "lightgreen", "darkgreen", "lightyellow",'black',
    "darkred", "lightpink", "skyblue", "gold", "violet", "indigo"
]
'''''''''
conn = sqlite3.connect('users.db')
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username BLOB NOT NULL,
        password BLOB NOT NULL,
        name BLOB NOT NULL
    )
    """)
conn.commit()
conn.close()
'''''''''
def check(c,password,username):
    print(password,username)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    username=sha_256.sha_256(username.encode())
    password=sha_256.sha_256(password.encode())
    query = f"SELECT * FROM users WHERE username = ? AND password = ? LIMIT 1"
    cursor.execute(query, (username,password))
    if cursor.fetchone():
        c.send(encryption.encryption("[SIGN_IN_RESPONSE]signed in").encode())
    else:
        c.send(encryption.encryption("[SIGN_IN_RESPONSE]not exists").encode())
    conn.close()
def insert(c, password, username, name):
    try:
        print(f"Attempting to insert: {username}, {name}")
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE username = ? LIMIT 1", (username,))
            if cursor.fetchone():
                print("User already exists.")
                c.send(encryption.encryption("[SIGN_UP_RESPONSE]exists").encode())
            else:
                password=sha_256.sha_256(password.encode())
                username=sha_256.sha_256(username.encode())
                name=sha_256.sha_256(name.encode())
                cursor.execute("INSERT INTO users (username, password, name) VALUES (?, ?, ?)",
                               (username, password, name))
                conn.commit()
                print("User inserted successfully.")
                c.send(encryption.encryption("[SIGN_UP_RESPONSE]inserted").encode())
    except Exception as e:
        print(f"Error in insert: {e}")
        traceback.print_exc()
        c.send(encryption.encryption("error").encode())
def transmit_media(client):
    print("media")
    data=client.recv(4096)
    while(not data.endswith(b'[END_OF_MEDIA]')):
        [c.send(data) for c in clients if c != client]
        data=client.recv(4096)
    [c.send(data) for c in clients if c != client]
    data=client.recv(1024)
    [c.send(data) for c in clients if c != client]
    print('end')
    stop.clear()
    msg=threading.Thread(target=handle_client,args=(client,))
    msg.start()
def handle_client(client):
    try:
        while True and not stop.is_set():
            msg = client.recv(1024)
            if not msg:
                break
            try:
                command = encryption.decryption(msg.decode())
            except:
                [c.send(msg) for c in clients if c !=client]
                continue
            print(command)
            if command.startswith('[SIGN_UP]'):
                insert(client,command.split('|||')[1],command.split('|||')[2],command.split('|||')[3])
            elif command.startswith('[SIGN_IN]'):
                check(client,command.split('|||')[1],command.split('|||')[2])
            elif msg.decode().startswith('[MEDIA]'):
                broadcast_message(client, msg)
                stop.set()
                transmit_media(client)
            else:
                broadcast_message(client, msg)
    except Exception as e:
        print(f"Error handling client: {e}")
        traceback.print_exc()
        if client in clients:
            clients.remove(client)
        client.close()
def broadcast_message(client, msg):
    try:
        print("the message is:",msg.decode())
        color = colors[clients.index(client) % len(colors)]
        for c in clients:
            if c != client:
                c.send('@@@~~~'.join([color,msg.decode()]).encode())
    except (BrokenPipeError, ConnectionResetError):
                print("A client disconnected during message broadcast.")
                if c in clients:
                    clients.remove(c)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip = socket.gethostbyname(socket.gethostname())
port = 5555
server.bind((ip, port))
server.listen()
print(f"Server started at {ip}:{port}")
clients = []
while True:
    client, address = server.accept()
    print(f"Connection established with {address}")
    clients.append(client)
    client.send(b"Welcome to the chat server!")
    threading.Thread(target=handle_client, args=(client,)).start()