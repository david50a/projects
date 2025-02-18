import math
def encryption(text):
    x = int(math.sqrt(len(text)))
    return "".join([chr((ord(i) + x) % 256) for i in text])
def decryption(text):
    x = int(math.sqrt(len(text)))
    return "".join([chr((ord(i) - x) % 256) for i in text])
# Example usage
'''''
original_text = "yoel"
encrypted = encryption(original_text)
decrypted = decryption(encrypted)
print("Original:", original_text)
print("Encrypted:", encrypted)
print("Decrypted:", decrypted)
'''''
