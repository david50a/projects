import os


class LCGRandom:
    def __init__(self, a, b):
        # a and b are secret
        self.seed = 0
        self.a = a
        self.b = b
        self.m = 256
        self.seed = self.NextByte()

    def NextByte(self):
        output = (self.a * self.seed + self.b) % self.m
        self.seed = output
        return output

    def GanerateByArray(self, n):
        return [self.NextByte() for i in range(n)]

    def encrypt(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                text = f.read()
                keypad = self.GanerateByArray(len(text))
                cipher = bytes((key ^ byte) for key, byte in zip(keypad, text))
            with open(path, 'wb') as f:
                f.write(cipher)
        else:
            return 'the file does not exist'

    def decrypt(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                msg = f.read()
                keypad = self.GanerateByArray(len(msg))
                cipher = bytes((key ^ byte) for key, byte in zip(keypad, msg))
            with open(path, 'wb') as f:
                f.write(cipher)

        else:
            return 'the file does not exist'
