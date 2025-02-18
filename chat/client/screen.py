import tkinter as tk
from tkinter import ttk, StringVar,LabelFrame
from tkinter import Frame, Label, Text,Canvas,END,filedialog,messagebox,Toplevel,WORD,Tk
from PIL import Image, ImageTk
import PIL
import socket
import threading
import communication_encyption
import encryption
import traceback
import re
import os
import image_steganography
import queue
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.padding import PKCS7
class AESMediaEncryptor:
    def __init__(self, password: str):
        self.password = password.encode()
        self.salt = os.urandom(16)
        self.key = self._derive_key(self.password, self.salt)

    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive a 32-byte AES key using PBKDF2 and SHA256."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(password)

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using AES CBC mode."""
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padder = PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data) + padder.finalize()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        return self.salt + iv + encrypted_data

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES CBC mode."""
        salt = encrypted_data[:16]
        iv = encrypted_data[16:32]
        encrypted_content = encrypted_data[32:]
        key = self._derive_key(self.password, salt)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(encrypted_content) + decryptor.finalize()
        unpadder = PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(decrypted_data) + unpadder.finalize()
media = None
stop_message = threading.Event()
stop_media = threading.Event()
running = True
message_queue =queue.Queue()
steganography=image_steganography.Image_steganography()
encryptor = AESMediaEncryptor("11c1s1dc1c515")
def sign_in(root):
    global password_in,username_in,answer
    root.destroy()
    # Create the main window
    root = tk.Tk()
    answer = StringVar()
    root.title("Vibewive")
    root.geometry("500x600")
    root.overrideredirect(True)  # Remove default title bar for a modern look
    root.config(bg="#f0f0f0")  # Light gray background

    # Custom title bar
    title_bar = tk.Frame(root, bg="#00ccff", relief="flat", height=30)
    title_bar.pack(side="top", fill="x")
    title_bar.bind("<B1-Motion>", move_window)
    title_label = tk.Label(title_bar, text="sign in", bg="#00ccff", fg="white", font=("Arial", 12))
    title_label.pack(side="left", padx=10)
    close_button = tk.Button(title_bar, text="X", bg="#f44336", fg="white", font=("Arial", 10), relief="flat", command=lambda: close_window(root))
    close_button.pack(side="right", padx=10)

    # Load and display the logo
    image = Image.open("logo.jpg")
    image = image.resize((100, 100))
    logo = ImageTk.PhotoImage(image)
    root.logo = logo  # Store reference to prevent garbage collection
    logo_label = tk.Label(root, image=logo)
    logo_label.pack(anchor="nw")

    # Content frame
    content_frame = tk.Frame(root, bg="#f0f0f0")
    content_frame.pack(expand=True, fill="both", padx=20, pady=20)

    # Modern Entry widget
    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Modern.TEntry",
        fieldbackground="#ffffff",
        foreground="#333333",
        padding=5,
        relief="flat",
    )
    style.map(
        "Modern.TEntry",
        fieldbackground=[("focus", "#99ebff")],
        bordercolor=[("focus", " #99ebff")],
    )
    tk.Label(content_frame, text="password", bg="#f0f0f0", fg="#333333", font=("Arial", 14)).pack()
    password_in = ttk.Entry(content_frame, style="Modern.TEntry", font=("Arial", 12),show="●")
    password_in.pack(pady=10, fill="x")
    tk.Label(content_frame, text="username", bg="#f0f0f0", fg="#333333", font=("Arial", 14)).pack()
    username_in = ttk.Entry(content_frame, style="Modern.TEntry", font=("Arial", 12))
    username_in.pack(pady=10, fill="x")

    # Modern Button widget
    style.configure(
        "Modern.TButton",
        font=("Arial", 12, "bold"),
        background="#3399ff",
        foreground="white",  # White text
        padding=10,
        borderwidth=0
    )
    style.map(
        "Modern.TButton",
        background=[
            ("active", "#ffcc00"),
            ("pressed", "#000000")
        ],
        foreground=[("disabled", "#0080ff")]  # Gray text when disabled
    )
    submit = ttk.Button(content_frame, text="Submit", style="Modern.TButton", command=lambda:check(root))
    submit.pack(pady=10)
    tk.Label(content_frame, text="if you have not an account press the sign up button", bg="#f0f0f0", fg="#333333", font=("Arial", 14)).pack()
    sign_in_button = ttk.Button(content_frame, text="sign up", style="Modern.TButton", command=lambda:sign_up(root))
    sign_in_button.pack(pady=10)
    response=tk.Label(master=root,textvariable=answer)
    response.pack()
    root.mainloop()
def sign_up(window=None):
    global answer_up,root
    global password_up,username_up,name
    # Create the main window
    if window != None:window.destroy()
    root = tk.Tk()
    answer_up= StringVar()
    root.title("Modern Window")
    root.geometry("500x600")
    root.overrideredirect(True)  # Remove default title bar for a modern look
    root.config(bg="#f0f0f0")  # Light gray background
    #Custom title bar
    title_bar = tk.Frame(root, bg="#00ccff", relief="flat", height=30)
    title_bar.pack(side="top", fill="x")
    title_bar.bind("<B1-Motion>", move_window)

    title_label = tk.Label(title_bar, text="sign up", bg="#00ccff", fg="white", font=("Arial", 12))
    title_label.pack(side="left", padx=10)

    close_button = tk.Button(title_bar, text="X", bg="#f44336", fg="white", font=("Arial", 10), relief="flat", command=lambda:close_window(root))
    close_button.pack(side="right", padx=10)
    image = Image.open("logo.jpg")
    image = image.resize((100, 100))
    logo = ImageTk.PhotoImage(image)
    logo_label = tk.Label(root, image=logo)
    logo_label.pack(anchor="nw")
    # Content frame
    content_frame = tk.Frame(root, bg="#f0f0f0")
    content_frame.pack(expand=True, fill="both", padx=20, pady=20)

    # Modern Entry widget
    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Modern.TEntry",
        fieldbackground="#ffffff",
        foreground="#333333",
        padding=5,
        relief="flat",
    )
    style.map(
        "Modern.TEntry",
        fieldbackground=[("focus", "#99ebff")],
        bordercolor=[("focus", " #99ebff")],
    )
    tk.Label(content_frame, text="password", bg="#f0f0f0", fg="#333333", font=("Arial", 14)).pack()
    password_up = ttk.Entry(content_frame, style="Modern.TEntry", font=("Arial", 12),show="●")
    password_up.pack(pady=10, fill="x")
    tk.Label(content_frame, text="username", bg="#f0f0f0", fg="#333333", font=("Arial", 14)).pack()
    username_up= ttk.Entry(content_frame, style="Modern.TEntry", font=("Arial", 12))
    username_up.pack(pady=10, fill="x")
    tk.Label(content_frame, text="name", bg="#f0f0f0", fg="#333333", font=("Arial", 14)).pack()
    name = ttk.Entry(content_frame, style="Modern.TEntry", font=("Arial", 12))
    name.pack(pady=10, fill="x")

    # Modern Button widget
    style.configure(
        "Modern.TButton",
        font=("Arial", 12, "bold"),
        background="#3399ff",
        foreground="white",    # White text
        padding=10,
        borderwidth=0
    )
    style.map(
        "Modern.TButton",
        background=[
            ("active", "#ffcc00"),
            ("pressed", "#000000")
        ],
        foreground=[("disabled", "#0080ff")]  # Gray text when disabled
    )

    submit = ttk.Button(content_frame, text="Submit", style="Modern.TButton",command=lambda:insert(root))
    submit.pack(pady=10)
    tk.Label(content_frame, text="if you have an account press the sign in button", bg="#f0f0f0", fg="#333333", font=("Arial", 14)).pack()
    sign_in_button = ttk.Button(content_frame, text="sign in", style="Modern.TButton", command=lambda: sign_in(root))
    sign_in_button.pack(pady=10)
    #Run the application
    response=tk.Label(root,textvariable=answer_up)
    response.pack()
    root.mainloop()
def add_message(message, sender=None, color=None):
    """Add a text message bubble to the chat."""
    bubble_frame = Frame(chat_frame, bg=color)  # Use chat_frame as the parent
    bubble_frame.pack(anchor="w" if sender == "user" else "e", pady=5, padx=10)
    bubble_color = "#00ccff" if sender == "user" else color
    text_color = "#ffffff" if sender == "user" else "#333333"
    bubble_label = Label(
        bubble_frame,
        text=message,
        bg=bubble_color,
        fg=text_color,
        font=("Arial", 12),
        wraplength=300,
        justify="left",
        padx=10,
        pady=5,
        bd=0,
    )
    bubble_label.pack()

    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1.0)

def open_image(file_path):
    """Open a full-size image in a new window."""
    img_window = Toplevel(window)
    img_window.title("Image Viewer")
    img = Image.open(file_path)
    img = ImageTk.PhotoImage(img)

    img_label = Label(img_window, image=img)
    img_label.image = img  # Keep a reference to prevent garbage collection
    img_label.pack()


def play_video(file_path):
    os.startfile(file_path)
def upload_media(sender="user", file_path=None):
    """Upload and display an image or video in the chat."""
    if not file_path:
        file_path = filedialog.askopenfilename(
            title="Select Media",
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.gif"),
                ("Video Files", "*.mp4;*.avi;*.mkv;*.mov"),
            ]
        )
    if not file_path:
        return  # Exit if no file is selected

    ext = os.path.splitext(file_path)[1].lower()
    bubble_frame = Frame(chat_frame, bg="#f0f0f0")
    bubble_frame.pack(anchor="w" if sender == "user" else 'e', pady=5, padx=10)

    if ext in [".png", ".jpg", ".jpeg", ".gif"]:
        try:
            img = Image.open(file_path)
            img.thumbnail((200, 200))  # Resize image for display
            img = ImageTk.PhotoImage(img)
            bubble_label = Label(bubble_frame, image=img, bg="#e6e6e6", bd=0)
            bubble_label.image = img  # Keep a reference to prevent garbage collection
            bubble_label.pack()
            bubble_label.bind("<Button-1>", lambda e: open_image(file_path))  # Bind click event
        except PIL.UnidentifiedImageError:
            print(f"Error: Could not open image file '{file_path}'. Unsupported format or corrupted file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    elif ext in [".mp4", ".avi", ".mkv", ".mov"]:
        video_label = Label(bubble_frame, text=f"Video: {os.path.basename(file_path)}", bg="#e6e6e6", fg="#333333",
                             font=("Arial", 12), padx=10, pady=5, bd=0)
        video_label.pack()

        play_button = ttk.Button(bubble_frame, text="Play Video", command=lambda: play_video(file_path))
        play_button.pack(anchor='w' if sender == 'user' else 'e', pady=5)

    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1.0)
    if(sender=='user'):
        send_media(file_path)
        client.send(b'[MEDIA]')
def send_media(path):
    try:
        filename = os.path.basename(path)
        with open(path, 'rb') as file:
            data=file.read()
        data=encryptor.encrypt_data(data)
        # Prepare metadata with unique delimiters
        info = "@@@@&&&&***".join([filename, str(len(data))])
        info, keys = communication_encyption.encryption(info)
        keys = "$$$$".join([str(k) for k in keys])
        msg = "@@@~~~".join(['[MEDIA]', info, encryption.encryption(keys)])
        client.send(msg.encode())
        client.sendall(data)
        print(f"Media size: {len(data)} bytes")
        # Send the file data in chunks

        # Send end-of-file marker separately to ensure correct transmission
        client.send(b'[END_OF_MEDIA]')
        print(f"Media sent: {filename}")
    except Exception as e:
        print(f"Error sending media: {e}")
        traceback.print_exc()
def receive_media(path, keys):
    print("Receiving media...")
    try:
        os.makedirs("media", exist_ok=True)
        keys = tuple(map(int, encryption.decryption(keys).split('$$$$')))
        info = communication_encyption.decryption(path, keys)
        filename, size = info.split('@@@@&&&&***')
        size = int(size)
        received_data = b''

        while len(received_data) < size:
            buffer = client.recv(min(4096, size - len(received_data)))
            if not buffer:
                break
            received_data += buffer

        data = encryptor.decrypt_data(received_data)
        save_path = os.path.join("media", filename)
        with open(save_path, 'wb') as file:
            file.write(data)

        eof_marker = client.recv(1024)
        if eof_marker.startswith(b'[END_OF_MEDIA]'):
            print("End-of-file marker received.")
            if eof_marker.endswith(b'[MEDIA]'):
                window.after(0, lambda: upload_media('other', save_path))  # Ensure UI update runs in the main thread
            else:
                add_message(f"you have received a steganography image with the name {filename}",'other','white')
        else:
            print("Error: EOF marker not received correctly.")

        print("Media received successfully.")
    except Exception as e:
        print(f"Error receiving media: {e}")
        traceback.print_exc()

    finally:
        stop_message.clear()
        msg = threading.Thread(target=receive_messages)
        msg.start()

def send_message():
    """Send a text message."""
    message = entry_msg.get().strip()
    if message:
        add_message(message, sender="user", color="#f0f0f0")
        entry_msg.delete(0, tk.END)
        msg, keys = communication_encyption.encryption(message)
        keys = [str(k) for k in keys]
        print("the keys are: ",keys)
        text_keys = "&".join(keys)
        client.send('@@@~~~'.join([msg,encryption.encryption(text_keys)]).encode())
        print("the message have been sent")

def open_encrypt_screen():
    """Open the screen for encrypting a message into an image."""

    def select_image():
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if image_path:
            try:
                img = Image.open(image_path)
                img = img.resize((700, 525))  # Resize for larger preview
                preview_image = ImageTk.PhotoImage(img)
                canvas.create_image(0, 0, anchor="nw", image=preview_image)
                canvas.image = preview_image  # Keep a reference
                canvas.image_path = image_path  # Store the selected image path
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def encrypt():
        message = message_entry.get("1.0", "end").strip()
        if not message:
            messagebox.showerror("Error", "Message cannot be empty!")
            return

        if hasattr(canvas, "image_path"):
            image_path = canvas.image_path
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")]
            )
            if save_path:
                try:
                    steganography.encrypt_text_in_image(image_path, message, target_path=save_path)
                    messagebox.showinfo("Success", f"Message encoded and saved to {save_path}")
                    send_media(save_path)
                    client.send(b'[steganography]')
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to encode message: {e}")
        else:
            messagebox.showerror("Error", "No image selected!")

    encrypt_screen = Toplevel()
    encrypt_screen.title("Steganography - Encrypt")
    encrypt_screen.geometry("800x600")
    encrypt_screen.config(bg="#f0f0f0")

    # Header with Logo
    header = Frame(encrypt_screen, bg="#00ccff", height=100)
    header.pack(fill="x", side="top")
    header_label = Label(header, text="Encrypt Message", bg="#00ccff", fg="white", font=("Arial", 16, "bold"))
    header_label.pack(side="left", padx=20)
    # Load and display the logo
    image = Image.open("logo.jpg")
    image = image.resize((100, 100))
    logo = ImageTk.PhotoImage(image)
    encrypt_screen.logo = logo  # Store reference to prevent garbage collection
    logo_label = tk.Label(encrypt_screen, image=logo)
    logo_label.pack(anchor="nw")

    # Main content frame
    content_frame = Frame(encrypt_screen, bg="#f0f0f0")
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Left panel for message entry
    left_frame = Frame(content_frame, bg="#f0f0f0")
    left_frame.pack(side="left", fill="y", padx=10, pady=10)

    Label(left_frame, text="Enter the message to encrypt:", font=("Arial", 12), bg="#f0f0f0").pack(anchor="w", pady=5)
    message_entry = Text(left_frame, width=50, height=15, font=("Arial", 12), wrap=WORD)
    message_entry.pack(padx=5, pady=5)

    # Right panel for image preview
    right_frame = Frame(content_frame, bg="#f0f0f0")
    right_frame.pack(side="right", fill="y", padx=10, pady=10)

    canvas = Canvas(right_frame, width=700, height=525, bg="gray")
    canvas.pack(pady=5)

    # Buttons
    button_frame = Frame(encrypt_screen, bg="#f0f0f0")
    button_frame.pack(anchor="center", side="bottom", pady=10)

    ttk.Button(button_frame, text="Select Image", command=select_image, style="Modern.TButton").pack(side="left",
                                                                                                     padx=5)
    ttk.Button(button_frame, text="Encrypt and Save", command=encrypt, style="Modern.TButton").pack(side="left", padx=5)

def open_decrypt_screen():
    """Open the screen for decrypting a message from an image."""

    def select_image():
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if image_path:
            try:
                img = Image.open(image_path)
                img = img.resize((500, 375))  # Resize for larger preview
                preview_image = ImageTk.PhotoImage(img)
                canvas.create_image(0, 0, anchor="nw", image=preview_image)
                canvas.image = preview_image  # Keep a reference
                canvas.image_path = image_path  # Store the selected image path
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def decrypt():
        if hasattr(canvas, "image_path"):
            image_path = canvas.image_path
            try:
                decoded_message = steganography.decrypt_text_in_image(image_path)
                if decoded_message:
                    # Update the text box with the decrypted message
                    msg_text.delete("1.0", END)
                    msg_text.insert(END, decoded_message)
                    messagebox.showinfo("Decoded Message", "Hidden message decoded successfully!")
                else:
                    messagebox.showwarning("No Message", "No hidden message found in the image!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to decode message: {e}")
        else:
            messagebox.showerror("Error", "No image selected!")

    decrypt_screen = Toplevel()
    decrypt_screen.title("Steganography - Decrypt")
    decrypt_screen.geometry("700x600")
    decrypt_screen.config(bg="#f0f0f0")

    # Header with Logo
    header = Frame(decrypt_screen, bg="#00ccff", height=50)
    header.pack(fill="x", side="top")
    header_label = Label(header, text="Decrypt Message", bg="#00ccff", fg="white", font=("Arial", 16, "bold"))
    header_label.pack(side="left", padx=20)
    # Load and display the logo
    image = Image.open("logo.jpg")
    image = image.resize((100, 100))
    logo = ImageTk.PhotoImage(image)
    decrypt_screen.logo = logo  # Store reference to prevent garbage collection
    logo_label = tk.Label(decrypt_screen, image=logo)
    logo_label.pack(anchor="nw")

    # Main content frame
    content_frame = Frame(decrypt_screen, bg="#f0f0f0")
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Image Preview Canvas
    canvas = Canvas(content_frame, width=500, height=375, bg="gray")
    canvas.pack(anchor="center", pady=10)

    # Text box to display decrypted message
    msg_frame = LabelFrame(content_frame, text="Decrypted Message", bg="#f0f0f0", font=("Arial", 12))
    msg_frame.pack(fill="both", expand=True, padx=10, pady=10)
    msg_text = Text(msg_frame, wrap=WORD, font=("Arial", 12), bg="#ffffff", fg="#000000", height=5)
    msg_text.pack(fill="both", expand=True, padx=5, pady=5)

    # Buttons
    button_frame = Frame(decrypt_screen, bg="#f0f0f0")
    button_frame.pack(anchor="center", side="bottom", pady=10)

    ttk.Button(button_frame, text="Select Image", command=select_image, style="Modern.TButton").pack(side="left",
                                                                                                     padx=5)
    ttk.Button(button_frame, text="Decrypt Message", command=decrypt, style="Modern.TButton").pack(side="left", padx=5)

def move_screen(event):
    window.geometry(f"+{event.x_root}+{event.y_root}")
def chat(root):
    global chat_canvas,chat_frame,window,entry_msg,text_label,thread,window
    root.destroy()

    # Main window setup
    window = Tk()
    window.geometry("500x700")
    window.title("app")
    window.config(bg="#f0f0f0")
    window.overrideredirect(True)  # Remove default title bar for a modern look
    title_bar = tk.Frame(window, bg="#00ccff", relief="flat", height=30)
    title_bar.pack(side="top", fill="x")
    title_bar.bind("<B1-Motion>", move_screen)

    title_label = tk.Label(title_bar, text="VibeWive", bg="#00ccff", fg="white", font=("Arial", 12),relief='flat')
    title_label.pack(side="left", padx=10)

    close_button = tk.Button(title_bar, text="X", bg="#f44336", fg="white", font=("Arial", 10), relief="flat",
                             command=lambda: close_window(window))  # Use 'window' instead of 'root'
    close_button.pack(side="right", padx=10)
    # Header
    image = Image.open("logo.jpg")
    image = image.resize((100, 100))
    logo = ImageTk.PhotoImage(image)
    window.logo=logo
    logo_label = tk.Label(window, image=logo)
    logo_label.pack(anchor="nw")

    # Chat canvas setup
    chat_canvas = Canvas(window, bg="#f0f0f0", bd=0, highlightthickness=0)
    chat_canvas.pack(fill="both", expand=True, padx=10, pady=5)

    # Frame inside canvas for chat bubbles
    chat_frame = Frame(chat_canvas, bg="#f0f0f0")
    chat_frame_window = chat_canvas.create_window((0, 0), window=chat_frame, anchor="nw")

    # Scrollbar
    scrollbar = ttk.Scrollbar(window, orient="vertical", command=chat_canvas.yview)
    scrollbar.pack(side="right", fill="y")
    chat_canvas.config(yscrollcommand=scrollbar.set)

    # Adjust scroll region
    def configure_scroll_region(event):
        chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))

    chat_frame.bind("<Configure>", configure_scroll_region)

    # Resize chat frame width based on canvas size
    def resize_width(event):
        chat_canvas.itemconfig(chat_frame_window, width=event.width)

    chat_canvas.bind("<Configure>", resize_width)

    # Bottom frame for message entry and buttons
    bottom_frame = Frame(window, bg="#f0f0f0")
    bottom_frame.pack(side="bottom", fill="x", pady=10, padx=10)

    # Subframe to hold buttons side by side
    button_frame = ttk.Frame(bottom_frame)
    button_frame.pack(anchor="center", side="bottom", pady=10)

    # Buttons within the button_frame
    send_button = ttk.Button(button_frame, text="Send", command=send_message, style="Modern.TButton")
    send_button.pack(side="left", padx=5)

    upload_button = ttk.Button(button_frame, text="Upload Media", command=lambda:upload_media('user',None), style="Modern.TButton")
    upload_button.pack(side="left", padx=5)
    steg_en_button = ttk.Button(button_frame, text="Steganography (encrypt)", command=open_encrypt_screen,
                                style="Modern.TButton")
    steg_en_button.pack(side="left", padx=5)
    steg_de_button = ttk.Button(button_frame, text="Steganography (decrypt)", style="Modern.TButton",
                                command=open_decrypt_screen)
    steg_de_button.pack()

    # Entry widget remains in the bottom_frame
    entry_msg = ttk.Entry(bottom_frame, width=40, font=("Arial", 12), style="Modern.TEntry")
    entry_msg.pack(anchor="center", side="bottom", pady=10)

    # Modern Button widget
    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Modern.TButton",
        font=("Arial", 12, "bold"),
        background="#3399ff",
        foreground="white",  # White text
        padding=10,
        borderwidth=0
    )
    style.map(
        "Modern.TButton",
        background=[
            ("active", "#ffcc00"),
            ("pressed", "#000000")
        ],
        foreground=[("disabled", "#0080ff")]  # Gray text when disabled
    )
    process_queue()
    thread = threading.Thread(target=receive_messages, daemon=True)
    thread.start()
    window.mainloop()
def move_window(event):
    root.geometry(f"+{event.x_root}+{event.y_root}")

def close_window(root):
    global running
    running = False
    root.destroy()
    client.close()
    thread.join()
def check(root):
    password=password_in.get()
    username=username_in.get()
    data = "|||".join(['[SIGN_IN]',password, username])
    client.send(encryption.encryption(data).encode())
    msg=client.recv(1024).decode()
    msg=encryption.decryption(msg)
    print(msg)
    if msg!='[SIGN_IN_RESPONSE]signed in':answer.set("the password or the username are incorrect")
    else:
        if not os.path.exists('media'):os.mkdir('media')
        chat(root=root)
def is_valid_password(password):
    patterns = [r'[A-Z]', r'[a-z]', r'\d', r'[^\w]']
    return all(re.search(p, password) for p in patterns) and len(password) >= 8
def validate_credentials(username, password):
    errors = []
    if len(username) < 4:
        errors.append("The username is too short (minimum 4 characters).")
    if not is_valid_password(password):
        errors.append("The password must include at least 8 characters, with uppercase,\n lowercase, digits, and special characters.")
    return errors
def insert(root):
    username = username_up.get()
    password = password_up.get()
    name_for_user = name.get()

    errors = validate_credentials(username, password)
    if errors:
        answer_up.set(" ".join(errors))
        return
    try:
        # Prepare and send the data
        data = "|||".join(['[SIGN_UP]', password, username, name_for_user])
        print("Sending to server:", data)
        client.send(encryption.encryption(data).encode())
        signup_response=client.recv(1024).decode()
        signup_response=encryption.decryption(signup_response)
        if signup_response == '[SIGN_UP_RESPONSE]inserted':
            if not os.path.exists("media"):
                os.mkdir("media")
            sign_in(root)
        elif signup_response == 'exists':
            answer_up.set("The username already exists.")
        else:
            answer_up.set("An error occurred. Please try again.")
    except Exception as e:
        print(f"Error during sign-up: {e}")
        answer_up.set("An error occurred. Please try again.")

def receive_messages():
    global running
    buffer = ""
    while running and not stop_message.is_set():
        try:
            data = client.recv(4096).decode()
            print(data)
            if not data:
                print("Server disconnected.")
                break
            buffer += data
            while "@@@~~~" in buffer :
                message= buffer.split("@@@~~~", )
                print(message)
                if message[1]=='[MEDIA]':
                    media_info, keys = message[-2],message[-1]
                    stop_message.set()
                    #media=threading.Thread(target=receive_media,args=(media_info,keys))
                    #media.start()
                    #media.join()
                    #stop_media.clear()
                    receive_media(media_info,keys)
                else:
                    print(data)
                    color, encrypted_msg, encrypted_keys = data.split('@@@~~~')
                    keys = encryption.decryption(encrypted_keys)
                    keys = tuple(map(int, keys.split('&')))
                    decrypted_msg = communication_encyption.decryption(encrypted_msg, keys)
                    print(f"Decrypted message: {decrypted_msg}, Color: {color}, Keys: {keys}")
                    message_queue.put((decrypted_msg, color))
                buffer=''
        except Exception as e:
            print("Error receiving messages:", e)
            traceback.print_exc()
            break

def process_queue():
    while not message_queue.empty():
        msg,color = message_queue.get()
        add_message(msg, sender="other", color=color)
    window.after(100, process_queue)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
server_ip = '10.0.0.152'
port = 5555
try:
    client.connect((server_ip, port))
    print(client.recv(1024).decode())
    sign_up()
except Exception as e:
    print(f"Error: {e}")
    client.close()