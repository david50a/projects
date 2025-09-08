import socket
import os

def get_file_content(filename):
    if(filename=='calculate-next'):
        return
    with open(filename, 'rb') as file:
        return file.read()

def get_file_extension(filename):
    print(filename)
    _, ext = os.path.splitext(filename)
    if(filename=='calculate-next'):
        print(_.lower())
        return _.lower()
    return ext.lower()

def generate_response(status, content_type, content):
    if isinstance(content, int) or isinstance(content,float):
        content=str(content).encode('utf-8')
    response_headers = f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n\r\n"
    response = response_headers.encode() + content
    return response

def handle_request(request):
    print(request.split())
    requested_file = request.split()[1][1:]  # Extract the requested file name from the request
    print(requested_file)
    if requested_file == '':
        requested_file = 'index.html'  # Default to serving index.html if no specific file is requested
    try:
        file_extension = get_file_extension(requested_file)

        if file_extension == '.html':
            content_type = 'text/html'
        elif file_extension == '.css':
            content_type = 'text/css'
        elif file_extension == '.js':
            content_type = 'application/javascript'
        elif file_extension in ['.jpg', '.jpeg']:
            content_type = 'image/jpeg'
        elif file_extension == '.png':
            content_type = 'image/png'
        else:
            if(requested_file=='calculate-next'):
                print('calculate')
                return generate_response('200 OK','text/plain',b'5')
            if(requested_file.startswith('calculate-next?num=')):
                num=requested_file[19:]
                print('num=' +num)
                if(num.isdigit()):
                    num=int(num)+1
                    return generate_response('200 OK','text/plain',num)
            if(requested_file.startswith('calculate-area?height=')):
                if(requested_file[22:requested_file.find('&')].isdigit()):
                    height=float(requested_file[22:requested_file.find('&')])
                width=requested_file[requested_file.find('&'):]
                if(width.startswith('&width=')):
                    if(width[7:].isdigit()):
                        width=float(width[7:])
                        return generate_response('200 OK','text/plain',((width*height)/2))
            print('error')
            return generate_response('404 Not Found', 'text/plain', b'404 Not Found')

        content = get_file_content(requested_file)
        return generate_response('200 OK', content_type, content)
    except FileNotFoundError:
        print('except')
        return generate_response('404 Not Found', 'text/plain', b'404 Not Found')

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    host = '0.0.0.0'
    port = 8080

    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server listening on http://{host}:{port}")

    while True:
        client_socket, client_address = server_socket.accept()
        request = client_socket.recv(1024).decode()

        response = handle_request(request)
        client_socket.sendall(response)
        client_socket.close()

if __name__ == "__main__":
    start_server()
