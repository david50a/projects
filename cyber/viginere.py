def encrypt(text,key):
    return translate(text,key,'encrypt')
def decrypt(cipher,key):
    return translate(cipher,key,'decrypt')
def translate(data,key,mode):
    mapping={chr(65+i):i for i in range(26)}
    letters=[chr(i+65) for i in range(26)]
    if mode=='encrypt':
        return ''.join([letters[(mapping[data[i]]+mapping[key[i%len(key)]])%26] for i in range(len(data))])
    if mode=='decrypt':
                return ''.join([letters[(mapping[data[i]]-mapping[key[i%len(key)]])%26] for i in range(len(data))])
cipher=encrypt('PROSPER','BOP')
print(cipher)
print(decrypt(cipher,'BOP'))
