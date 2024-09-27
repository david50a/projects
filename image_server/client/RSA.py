import random
import math
import os
class RSAEncryption:
    def __init__(self):
        self.parameters = self.initialize_parameters()

    @staticmethod
    def is_prime(number):
        if number < 2:
            return False
        for i in range(2, number // 2 + 1):
            if number % i == 0:
                return False
        return True

    def initialize_parameters(self):
        if os.path.exists('parameters.txt'):
            with open('parameters.txt', 'r') as f:
                parameter = f.read()
                parameters = parameter.split('\n')
        else:
            p, q = self.generate_prime(1000, 5000), self.generate_prime(1000, 5000)
            while p == q:
                q = self.generate_prime(1000, 5000)
            n = p * q
            phi_n = (p - 1) * (q - 1)
            e = random.randint(3, phi_n - 1)
            while math.gcd(e, phi_n) != 1:
                e = random.randint(3, phi_n)
            d = self.mod_inverse(e, phi_n)
            with open(r'parameters.txt', 'w') as file:
                file.write(str(p) + '\n')
                file.write(str(q) + '\n')
                file.write(str(n) + '\n')
                file.write(str(phi_n) + '\n')
                file.write(str(e) + '\n')
                file.write(str(d))
            parameters = [str(p), str(q), str(n), str(phi_n), str(e), str(d)]
        return parameters

    @staticmethod
    def generate_prime(min_value, max_value):
        prime = random.randint(min_value, max_value)
        while not RSAEncryption.is_prime(prime):
            prime = random.randint(min_value, max_value)
        return prime

    @staticmethod
    def mod_inverse(e, phi):
        for d in range(3, phi):
            if (d * e) % phi == 1:
                return d
        raise ValueError("mod_inverse does not exist")

    def encrypt(self, message, target_name):
        n = int(self.parameters[2])
        e = int(self.parameters[4])
        message = [ord(c) for c in message]
        ciphertext = [pow(c, e, n) for c in message]
        with open(target_name, 'w') as f:
            f.write(' '.join([str(c) for c in ciphertext]))
        with open(target_name, 'r') as f:
            ciphertext = f.read()
        return ciphertext

    def decrypt(self, ciphertext, target_name):
        print('The process is starting')
        d = int(self.parameters[5])
        n = int(self.parameters[2])
        with open(target_name, 'w') as f:
            f.write(ciphertext)
        ciphertext = ciphertext.split(' ')
        for i in range(len(ciphertext)):
            if ciphertext[i] == '' or ciphertext[i] == ' ':
                if i == len(ciphertext) - 1:
                    ciphertext = ciphertext[:i]
                else:
                    ciphertext = ciphertext[:i] + ciphertext[i + 1:]
        ciphertext = [int(i) for i in ciphertext]
        message_encoded = [pow(ch, d, n) for ch in ciphertext]
        message_encoded = [str(i) for i in message_encoded]
        with open(target_name, 'w') as f:
            f.write(' '.join(message_encoded))
        with open(target_name, 'w') as f:
            message_encoded = [int(i) for i in message_encoded]
            f.write(''.join(chr(ch) for ch in message_encoded))
        print('Finish the process')
        return ''.join(chr(ch) for ch in message_encoded)


#Example Usage:
#rsa = RSAEncryption()
# message = "Hello, world!"
# ciphertext = rsa.encrypt(message, "encrypted.txt")
# decrypted_message = rsa.decrypt(ciphertext, "decrypted.txt")
