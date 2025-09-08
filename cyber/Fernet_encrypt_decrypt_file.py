from cryptography.fernet import Fernet
def encrypt():
    key=Fernet.generate_key()
    encrypter=Fernet(key)
    with open('fernet_key.txt','wb')as f:
        f.write(key)
    with open('pic.jpg','rb')as f:
        data=f.read()
        with open('enc_pic.jpg','wb')as f:
            encrypted_data=encrypter.encrypt(data)
            f.write(encrypted_data)
def decrypt():
    with open('Fernet_key.txt','r')as f:
        key=f.read()
    decrypter=Fernet(key)
    with open('enc_pic.jpg','rb')as f:
        data=f.read()
    with open('dec_pic.jpg','wb')as f:
        decrypted_data=decrypter.decrypt(data)
        f.write(decrypted_data)
encrypt()
decrypt()