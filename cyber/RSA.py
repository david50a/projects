import random
import math
import os


class RSAEncryptorDecryptor:
    def __init__(self):
        self.parameters = self.initialize_parameters()

    @staticmethod
    def is_prime(number):
        if number <= 1:
            return False
        if number <= 3:
            return True
        if number % 2 == 0 or number % 3 == 0:
            return False
        i = 5
        while i * i <= number:
            if number % i == 0 or number % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def initialize_parameters():
        if not os.path.exists('parameters.txt') or os.path.getsize('parameters.txt') == 0:
            p = RSAEncryptorDecryptor.generate_prime(1000, 5000)
            q = RSAEncryptorDecryptor.generate_prime(1000, 5000)
            while p == q:
                q = RSAEncryptorDecryptor.generate_prime(1000, 5000)
            n = p * q
            phi_n = (p - 1) * (q - 1)
            e = random.randint(3, phi_n - 1)
            while math.gcd(e, phi_n) != 1:
                e = random.randint(3, phi_n - 1)
            d = RSAEncryptorDecryptor.mod_inverse(e, phi_n)
            with open('parameters.txt', 'w') as file:
                file.write(f"{p}\n{q}\n{n}\n{phi_n}\n{e}\n{d}\n")
        with open('parameters.txt', 'r') as file:
            parameters = file.read().strip().split('\n')
        return list(map(int, parameters))

    @staticmethod
    def generate_prime(min_value, max_value):
        prime = random.randint(min_value, max_value)
        while not RSAEncryptorDecryptor.is_prime(prime):
            prime = random.randint(min_value, max_value)
        return prime

    @staticmethod
    def mod_inverse(e, phi):
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, y = extended_gcd(e, phi)
        if gcd != 1:
            raise ValueError("mod_inverse does not exist")
        return x % phi

    def encrypt(self, data):
        n, e = self.parameters[2], self.parameters[4]
        ciphertext = [pow(c, e, n) for c in data]
        return ' '.join(map(str, ciphertext))

    def decrypt(self, ciphertext):
        d, n = self.parameters[5], self.parameters[2]
        ciphertext = list(map(int, ciphertext.split()))
        message_encoded = [pow(ch, d, n) for ch in ciphertext]
        return b''.join(bytes(ch) for ch in message_encoded)

# Example usage:
# rsa = RSAEncryptorDecryptor()
# encrypted_message = rsa.encrypt("Hello World!")
# print("Encrypted:", encrypted_message)
# decrypted_message = rsa.decrypt(encrypted_message)
# print("Decrypted:", decrypted_message)
rsa=RSAEncryptorDecryptor()
en=rsa.encrypt(b'meir')
print(en)
de=rsa.decrypt(en)
print(de)