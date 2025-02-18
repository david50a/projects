from datetime import datetime
import random

def generate_keys(text):
    num = str(datetime.now()).split(" ")[0].split("-")
    summary = [int(i) for i in num]
    key1 = (sum(summary) * summary[2] / summary[0]) ** summary[1]
    key2 = len(text) * sum(ord(c) for c in text)
    key3 = (key1 + key2) % 256
    return int(key1) % 256, int(key2) % 256, int(key3) % 256

def shuffle_text(text, key):
    random.seed(key)
    text = list(text)
    random.shuffle(text)
    return "".join(text)

def unshuffle_text(text, key):
    random.seed(key)
    indices = list(range(len(text)))
    shuffled_indices = indices[:]
    random.shuffle(shuffled_indices)
    unshuffled = [None] * len(text)
    for i, shuffled_index in enumerate(shuffled_indices):
        unshuffled[shuffled_index] = text[i]
    return "".join(unshuffled)

def encryption(text):
    # Reverse the text
    text = text[::-1]
    # Generate keys
    key1, key2, key3 = generate_keys(text)
    # Shuffle the text
    shuffled_text = shuffle_text(text, key3)
    # Encrypt the text using XOR with two keys
    encrypted = "".join([chr(((ord(c) ^ key1) + key2) % 256) for c in shuffled_text])
    return encrypted, (key1, key2, key3)

def decryption(encrypted_text, keys):
    key1, key2, key3 = keys
    # Decrypt using XOR with two keys
    decrypted = "".join([chr(((ord(c) - key2) ^ key1) % 256) for c in encrypted_text])
    # Unshuffle the text
    unshuffled_text = unshuffle_text(decrypted, key3)
    # Reverse the text back
    original_text = unshuffled_text[::-1]
    return original_text

# Example usage
'''''
text = "Hello, World!"
encrypted_text, keys = encryption(text)
print("Encrypted:", encrypted_text)
decrypted_text = decryption(encrypted_text, keys)
print("Decrypted:", decrypted_text)
'''''
