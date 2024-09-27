import socket
import os
from RSA import RSAEncryption
import cipher
import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab
import PIL.Image
import PIL.ImageTk
import cv2
from ttkthemes import ThemedTk
from tkinter import filedialog
from ascii import ASCIIArtConverter
import threading

class VideoApp:
    def __init__(self, root, video_source=0, width=400, height=300, tkinter_style="default"):
        self.root = root
        self.root.title("Video Player")
        self.frame = None
        # Open a connection to the webcam
        self.cap = cv2.VideoCapture(video_source)

        # Set the desired width and height for the displayed video
        self.width = width
        self.height = height

        # Create a Style and apply the theme to it
        style = ttk.Style()
        style.theme_use(tkinter_style)

        # Create a Frame to hold the Label and set its background color
        frame = ttk.Frame(root, style="TFrame")  # Use the "TFrame" style from the theme
        frame.pack(expand=tk.YES, fill=tk.BOTH)  # Add pady for vertical padding

        # Create a Label to display the video
        self.label = ttk.Label(frame, style="TLabel")  # Use the "TLabel" style from the theme
        self.label.pack(expand=tk.YES, fill=tk.BOTH)  # Add pady for vertical padding
        # Initialize the GUI
        self.update()

        # Run the Tkinter main loop
        root.mainloop()

    def update(self):
        # Read a frame from the camera
        ret, self.frame = self.cap.read()

        if ret:
            # Resize the frame to the desired width and height
            self.frame = cv2.resize(self.frame, (self.width, self.height))

            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a PhotoImage
            photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb_frame))

            # Update the Label with the new frame
            self.label.configure(image=photo)
            self.label.image = photo
            self.label.place(x=200, y=10)

        # Call the update method again after a delay (e.g., 33 milliseconds for ~30 fps)
        self.root.after(33, self.update)


    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


def process_image(image_path, client, b1, b2):
    client.send(b"[image]")
    b1.configure(state='disabled')
    b2.configure(state='disabled')
    output_string.set("Please wait until the ending of the process")
    # Encrypt the image
    convertor = ASCIIArtConverter(image_path)
    ascii_art, colors = convertor.image_to_ascii()
    convertor.create_data_files(colors, ascii_art)

    with open('ascii.txt', 'r') as f:
        da = f.read()
    try:
        rsa.encrypt(da, 'encrypted.txt')  # Ensure the data is in bytes
        send_ascii()

        if client.recv(1024).decode() == 'the ascii file arrived':
            send_color()

        caption()
    except:
        print("The client disconnected")

def send_image(image_path):
    # Create a new thread to process the image
    thread = threading.Thread(target=process_image, args=(image_path, client, b1, b2))
    thread.start()

def send_ascii():
    buffer_size = 0
    ascii_size = os.path.getsize('encrypted.txt')
    client.send(str(ascii_size).encode())
    with open('encrypted.txt', 'rb') as ascii_file:
        while buffer_size < ascii_size:
            buffer_ascii = ascii_file.read(1024)
            buffer_size += len(buffer_ascii)
            client.send(buffer_ascii)


def send_color():
    passed = 0
    file = open("colors.txt", 'r')
    rsa.encrypt(file.read(), 'colors.txt')
    colors_size = os.path.getsize('colors.txt')
    client.send(str(colors_size).encode())
    with open('colors.txt', 'r', encoding="latin1") as colors_file:
        while passed < colors_size:
            buffer_color = colors_file.read(1024)
            passed += len(buffer_color)
            client.send(buffer_color.encode())


def take_image():
    bbox = (250, 40, 1500, 980)
    region_screenshot = ImageGrab.grab(bbox)

    # Save the screenshot as an image file
    region_screenshot.save("output.png")
    send_image("output.png")


def caption():
    with open('p.txt', 'wb') as f:
        f.write(client.recv(1024))
    c.decrypt('p.txt')
    with open('p.txt', 'r') as caption:
        caption = caption.read()
    output_string.set(caption)
    b1.configure(state='normal')
    b2.configure(state='normal')


def choose_image():
    # Open a file dialog for selecting an image
    file_path = filedialog.askopenfilename(filetypes=(("PNG file", '*.png'), ('JPG file', '*.jpg')))

    if file_path:
        send_image(file_path)


def finish():
    client.send(b'[quit]')
    client.close()
    root.destroy()
    os.remove('parameters.txt')


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 555))
print(client.recv(1024).decode())
size = int(client.recv(1024).decode())
c = cipher.LCGRandom(101, 401)
total = 0
with open("parameters.txt", 'wb') as file:
    while total < size:
        data = client.recv(1024)
        file.write(data)
        total += len(data)
c.decrypt('parameters.txt')
rsa = RSAEncryption()

root = ThemedTk()
frame = ttk.Frame(master=root)
output_string = tk.StringVar(value='Output')
output_label = ttk.Label(master=root, wraplength=1000, justify='left', textvariable=output_string)
output_label.pack(side=tk.BOTTOM)
frame.pack(side=tk.RIGHT)
b1 = ttk.Button(master=frame, text="Send the picture", command=take_image)
b2 = ttk.Button(master=frame, text="import and send an image", command=choose_image)
b3 = ttk.Button(master=frame, text='exit', command=finish)
b1.pack(pady=20)
b2.pack(pady=10)
b3.pack(pady=30)
app = VideoApp(root, tkinter_style="equilux", width=1000, height=750)
client.close()
