class AES:
    def __init__(self, key):
        if len(key) != 16:
            raise ValueError("Key must be 16 bytes")
        self.Nb = 4
        self.Nk = 4
        self.Nr = 10
        self.round_keys = []
        self._key_expansion(key)
    
    # S-box
    _sbox = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0xdc, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d,
        0x9e, 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28,
        0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb
    ]
    
    # Inverse S-box
    _inv_sbox = [
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
    ]
    
    # Round constants
    _Rcon = [0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
    
    def _key_expansion(self, key):
        """Expand the key into round keys"""
        self.round_keys = [0] * (4 * self.Nb * (self.Nr + 1))
        
        # Copy the key into the first round key
        for i in range(self.Nk):
            self.round_keys[i*4] = key[i*4]
            self.round_keys[i*4+1] = key[i*4+1]
            self.round_keys[i*4+2] = key[i*4+2]
            self.round_keys[i*4+3] = key[i*4+3]
        
        # All other round keys are found from the previous round keys
        for i in range(self.Nk, self.Nb * (self.Nr + 1)):
            k = (i - 1) * 4
            temp = [
                self.round_keys[k],
                self.round_keys[k + 1],
                self.round_keys[k + 2],
                self.round_keys[k + 3]
            ]
            
            if i % self.Nk == 0:
                # Rotate word
                temp = [temp[1], temp[2], temp[3], temp[0]]
                
                # SubBytes
                temp = [self._sbox[b] for b in temp]
                
                temp[0] ^= self._Rcon[i // self.Nk]
            
            j = i * 4
            k = (i - self.Nk) * 4
            self.round_keys[j] = self.round_keys[k] ^ temp[0]
            self.round_keys[j + 1] = self.round_keys[k + 1] ^ temp[1]
            self.round_keys[j + 2] = self.round_keys[k + 2] ^ temp[2]
            self.round_keys[j + 3] = self.round_keys[k + 3] ^ temp[3]
    
    def _add_round_key(self, state, round_num):
        """Add round key to state"""
        for i in range(16):
            state[i] ^= self.round_keys[(round_num * self.Nb * 4) + i]
    
    def _sub_bytes(self, state):
        """Apply S-box substitution"""
        for i in range(16):
            state[i] = self._sbox[state[i]]
    
    def _inv_sub_bytes(self, state):
        """Apply inverse S-box substitution"""
        for i in range(16):
            state[i] = self._inv_sbox[state[i]]
    
    def _shift_rows(self, state):
        """Shift rows transformation"""
        # Rotate first row 1 columns to left
        temp = state[0 * 4 + 1]
        state[0 * 4 + 1] = state[1 * 4 + 1]
        state[1 * 4 + 1] = state[2 * 4 + 1]
        state[2 * 4 + 1] = state[3 * 4 + 1]
        state[3 * 4 + 1] = temp
        
        # Rotate second row 2 columns to left
        temp = state[0 * 4 + 2]
        state[0 * 4 + 2] = state[2 * 4 + 2]
        state[2 * 4 + 2] = temp
        
        temp = state[1 * 4 + 2]
        state[1 * 4 + 2] = state[3 * 4 + 2]
        state[3 * 4 + 2] = temp
        
        # Rotate third row 3 columns to left
        temp = state[0 * 4 + 3]
        state[0 * 4 + 3] = state[3 * 4 + 3]
        state[3 * 4 + 3] = state[2 * 4 + 3]
        state[2 * 4 + 3] = state[1 * 4 + 3]
        state[1 * 4 + 3] = temp
    
    def _inv_shift_rows(self, state):
        """Inverse shift rows transformation"""
        # Rotate first row 1 columns to right
        temp = state[3 * 4 + 1]
        state[3 * 4 + 1] = state[2 * 4 + 1]
        state[2 * 4 + 1] = state[1 * 4 + 1]
        state[1 * 4 + 1] = state[0 * 4 + 1]
        state[0 * 4 + 1] = temp
        
        # Rotate second row 2 columns to right
        temp = state[0 * 4 + 2]
        state[0 * 4 + 2] = state[2 * 4 + 2]
        state[2 * 4 + 2] = temp
        
        temp = state[1 * 4 + 2]
        state[1 * 4 + 2] = state[3 * 4 + 2]
        state[3 * 4 + 2] = temp
        
        # Rotate third row 3 columns to right
        temp = state[0 * 4 + 3]
        state[0 * 4 + 3] = state[1 * 4 + 3]
        state[1 * 4 + 3] = state[2 * 4 + 3]
        state[2 * 4 + 3] = state[3 * 4 + 3]
        state[3 * 4 + 3] = temp
    
    def _xtime(self, x):
        """Multiplication by 2 in GF(2^8)"""
        return ((x << 1) ^ (((x >> 7) & 1) * 0x1b)) & 0xFF
    
    def _mix_columns(self, state):
        """Mix columns transformation"""
        for i in range(4):
            t = state[i * 4 + 0]
            Tmp = state[i * 4 + 0] ^ state[i * 4 + 1] ^ state[i * 4 + 2] ^ state[i * 4 + 3]
            
            Tm = state[i * 4 + 0] ^ state[i * 4 + 1]
            Tm = self._xtime(Tm)
            state[i * 4 + 0] ^= Tm ^ Tmp
            
            Tm = state[i * 4 + 1] ^ state[i * 4 + 2]
            Tm = self._xtime(Tm)
            state[i * 4 + 1] ^= Tm ^ Tmp
            
            Tm = state[i * 4 + 2] ^ state[i * 4 + 3]
            Tm = self._xtime(Tm)
            state[i * 4 + 2] ^= Tm ^ Tmp
            
            Tm = state[i * 4 + 3] ^ t
            Tm = self._xtime(Tm)
            state[i * 4 + 3] ^= Tm ^ Tmp
    
    def _multiply(self, x, y):
        """Multiplication in GF(2^8)"""
        return (((y & 1) * x) ^
                ((y >> 1 & 1) * self._xtime(x)) ^
                ((y >> 2 & 1) * self._xtime(self._xtime(x))) ^
                ((y >> 3 & 1) * self._xtime(self._xtime(self._xtime(x)))) ^
                ((y >> 4 & 1) * self._xtime(self._xtime(self._xtime(self._xtime(x)))))) & 0xFF
    
    def _inv_mix_columns(self, state):
        """Inverse mix columns transformation"""
        for i in range(4):
            a = state[i * 4 + 0]
            b = state[i * 4 + 1]
            c = state[i * 4 + 2]
            d = state[i * 4 + 3]
            
            state[i * 4 + 0] = (self._multiply(a, 0x0e) ^ self._multiply(b, 0x0b) ^ 
                               self._multiply(c, 0x0d) ^ self._multiply(d, 0x09))
            state[i * 4 + 1] = (self._multiply(a, 0x09) ^ self._multiply(b, 0x0e) ^ 
                               self._multiply(c, 0x0b) ^ self._multiply(d, 0x0d))
            state[i * 4 + 2] = (self._multiply(a, 0x0d) ^ self._multiply(b, 0x09) ^ 
                               self._multiply(c, 0x0e) ^ self._multiply(d, 0x0b))
            state[i * 4 + 3] = (self._multiply(a, 0x0b) ^ self._multiply(b, 0x0d) ^ 
                               self._multiply(c, 0x09) ^ self._multiply(d, 0x0e))
    
    def _cipher(self, state):
        """AES cipher"""
        self._add_round_key(state, 0)
        
        for round_num in range(1, self.Nr):
            self._sub_bytes(state)
            self._shift_rows(state)
            self._mix_columns(state)
            self._add_round_key(state, round_num)
        
        self._sub_bytes(state)
        self._shift_rows(state)
        self._add_round_key(state, self.Nr)
    
    def _inv_cipher(self, state):
        """AES inverse cipher"""
        self._add_round_key(state, self.Nr)
        
        for round_num in range(self.Nr - 1, 0, -1):
            self._inv_shift_rows(state)
            self._inv_sub_bytes(state)
            self._add_round_key(state, round_num)
            self._inv_mix_columns(state)
        
        self._inv_shift_rows(state)
        self._inv_sub_bytes(state)
        self._add_round_key(state, 0)
    
    def _pkcs7_pad(self, data):
        """Apply PKCS7 padding"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, list):
            data = bytes(data)
        
        pad_len = 16 - (len(data) % 16)
        padded = bytearray(data)
        for _ in range(pad_len):
            padded.append(pad_len)
        return padded
    
    def _pkcs7_unpad(self, data):
        """Remove PKCS7 padding"""
        if not data:
            return data
        
        pad_len = data[-1]
        if pad_len < 1 or pad_len > 16:
            return data  # Invalid padding
        
        # Verify padding
        for i in range(len(data) - pad_len, len(data)):
            if data[i] != pad_len:
                return data  # Invalid padding
        
        return data[:-pad_len]
    
    def encrypt(self, plaintext):
        """Encrypt plaintext"""
        padded = self._pkcs7_pad(plaintext)
        encrypted = bytearray()
        
        for i in range(0, len(padded), 16):
            block = list(padded[i:i+16])
            self._cipher(block)
            encrypted.extend(block)
        
        return bytes(encrypted)
    
    def decrypt(self, ciphertext):
        """Decrypt ciphertext"""
        if len(ciphertext) % 16 != 0:
            raise ValueError("Ciphertext length must be multiple of 16")
        
        decrypted = bytearray()
        
        for i in range(0, len(ciphertext), 16):
            block = list(ciphertext[i:i+16])
            self._inv_cipher(block)
            decrypted.extend(block)
        
        return bytes(self._pkcs7_unpad(decrypted))
    
    def to_hex(self, data):
        """Convert bytes to hex string"""
        return data.hex()
    
    def from_hex(self, hex_string):
        """Convert hex string to bytes"""
        return bytes.fromhex(hex_string)


# Example usage
if __name__ == "__main__":
    try:
        # 128-bit key (16 bytes)
        key = bytes([
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
        ])
        
        aes = AES(key)
        
        # Test with text
        text = "Hello, AES encryption! This is a test message."
        print(f"Original text: {text}")
        
        # Encrypt
        encrypted = aes.encrypt(text)
        print(f"Encrypted (hex): {aes.to_hex(encrypted)}")
        
        # Decrypt
        decrypted = aes.decrypt(encrypted)
        decrypted_text = decrypted.decode('utf-8')
        print(f"Decrypted text: {decrypted_text}")
        
        # Test with binary data
        binary_data = bytes([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0xFF, 0xFE, 0xFD])
        print(f"\nOriginal binary (hex): {aes.to_hex(binary_data)}")
        
        encrypted_binary = aes.encrypt(binary_data)
        print(f"Encrypted binary (hex): {aes.to_hex(encrypted_binary)}")
        
        decrypted_binary = aes.decrypt(encrypted_binary)
        print(f"Decrypted binary (hex): {aes.to_hex(decrypted_binary)}")
        
        # Test with data not divisible by 16
        short_data = b"Short!"
        print(f"\nOriginal short data: {short_data}")
        encrypted_short = aes.encrypt(short_data)
        print(f"Encrypted short (hex): {aes.to_hex(encrypted_short)}")
        decrypted_short = aes.decrypt(encrypted_short)
        print(f"Decrypted short: {decrypted_short}")
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
