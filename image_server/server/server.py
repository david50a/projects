import os
import socket
from ascii import ASCIIArtConverter
from RSA import RSAEncryptorDecryptor
import cipher
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

rsa = RSAEncryptorDecryptor()
ascii = ASCIIArtConverter
c = cipher.LCGRandom(101, 401)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', 555))
print("The server is running")
server.listen()
client, address = server.accept()
client.send("You are connected!!!".encode())
print("The client connected")

with open('encrypted_parameters.txt', 'w') as f, open('parameters.txt', 'r') as file:
    f.write(file.read())
c.encrypt('encrypted_parameters.txt')
with open('encrypted_parameters.txt', 'rb') as file:
    size = os.path.getsize('encrypted_parameters.txt')
    client.send(str(size).encode())
    total = 0
    while total < size:
        buffer = file.read(1024)
        total += len(buffer)
        client.send(buffer)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {'max_length': max_length, "num_beams": num_beams}


def get_image():
    try:
        print('the process is starting')
        size = int(client.recv(1024).decode())
        data = b''
        while len(data) < size:
            msg = client.recv(1024)
            data += msg
        ascii_data = rsa.decrypt(data.decode())
        client.send("the ascii file arrived".encode())
        size = int(client.recv(1024).decode())
        colors_data = b''
        while size > len(colors_data):
            msg = client.recv(1024)
            colors_data += msg
        print('finish the transfer')
        colors_data = rsa.decrypt(colors_data.decode())
        with open('output_colors.txt', 'w') as f:
            f.write(colors_data)
        colors_data = ascii.create_color_array('output_colors.txt')
        ascii.ascii_to_image(ascii_data, colors_data, 'output.jpg')
    except:
        print("The client has been disconnected.")


def caption(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    print(preds[0])
    return str(preds[0])


while True:
    try:
        msg = client.recv(1024)
    except:
        print("The client has been disconnected.")
        break
    if msg == b"[image]":
        get_image()
        with open('p.txt', 'w') as f:
            caption_text = caption(['output.jpg'])
            f.write(caption_text)
        c.encrypt('p.txt')
        with open('p.txt', 'rb') as caption_file:
            encrypted_caption = caption_file.read()
        client.send(encrypted_caption)
    if msg == b"[quit]":
        break
os.remove('parameters.txt')
os.remove('encrypted_parameters.txt')
server.close()
