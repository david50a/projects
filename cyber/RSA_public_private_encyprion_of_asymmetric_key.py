from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES,PKCS1_OAEP
import base64
#generate keys
key=RSA.generate(2048)
private_key=key.export_key()
public_key=key.public_key().export_key()
with open('private.pem','wb')as f:
    f.write(private_key)
with open('public.pem','wb')as f:
    f.write(public_key)
#Encrypt the RSA keys
def encrypt_keys():
    print('> Encryption')
    public_key=RSA.import_key(open('public.pem').read())
    with open('fernet_key.txt','rb')as f:
        fernet_key=f.read()
    public_encrypter=PKCS1_OAEP.new(public_key)
    with open('enc_fernet_key.txt','wb')as f:
        enc_fernet_key=public_encrypter.encrypt(fernet_key)
        f.write(enc_fernet_key)
    print('Public key: ',public_key)
    print('Fernet key: ',fernet_key)
    print('Public encrypter',public_encrypter)
    print('Encrypted fernet key: ',enc_fernet_key)
    print("> Encryption complicated")
#Decrypt the RSA keys
def decrypt_keys():
    print('> Decryption')
    with open('enc_fernet_key.txt','rb')as f:
        enc_fernet_key=f.read()
    private_key=RSA.import_key(open('private.pem',).read())
    private_decrypter=PKCS1_OAEP.new(private_key)
    dec_fernet_key=private_decrypter.decrypt(enc_fernet_key)
    with open('enc_fernet_key.txt','wb')as f:
        f.write(dec_fernet_key)
    print('Private key: ',private_key)
    print('Private decrypter',private_decrypter)
    print('Decrypted fernet key: ',dec_fernet_key)
    print("> Decryption complicated")