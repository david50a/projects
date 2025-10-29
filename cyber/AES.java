import java.util.Arrays;
import java.nio.charset.StandardCharsets;

public class AES {
    private static final int Nb = 4;
    private static final int Nk = 4;
    private static final int Nr = 10;
    
    // S-box
    private static final int[] sbox = {
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
    };
    
    // Inverse S-box
    private static final int[] invSbox = {
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
    };
    
    // Round constants
    private static final int[] Rcon = {
        0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
    };
    
    private int[] roundKeys = new int[176]; // 4 * Nb * (Nr + 1)
    
    public AES(byte[] key) {
        if (key.length != 16) {
            throw new IllegalArgumentException("Key must be 16 bytes");
        }
        keyExpansion(key);
    }
    
    private void keyExpansion(byte[] key) {
        int[] tempa = new int[4];
        int i, j, k;
        
        // Copy the key into the first round key
        for (i = 0; i < Nk; i++) {
            roundKeys[i * 4] = key[i * 4] & 0xFF;
            roundKeys[i * 4 + 1] = key[i * 4 + 1] & 0xFF;
            roundKeys[i * 4 + 2] = key[i * 4 + 2] & 0xFF;
            roundKeys[i * 4 + 3] = key[i * 4 + 3] & 0xFF;
        }
        
        // All other round keys are found from the previous round keys
        for (i = Nk; i < Nb * (Nr + 1); i++) {
            k = (i - 1) * 4;
            tempa[0] = roundKeys[k];
            tempa[1] = roundKeys[k + 1];
            tempa[2] = roundKeys[k + 2];
            tempa[3] = roundKeys[k + 3];
            
            if (i % Nk == 0) {
                // Rotate word
                int u8tmp = tempa[0];
                tempa[0] = tempa[1];
                tempa[1] = tempa[2];
                tempa[2] = tempa[3];
                tempa[3] = u8tmp;
                
                // SubBytes
                tempa[0] = sbox[tempa[0]];
                tempa[1] = sbox[tempa[1]];
                tempa[2] = sbox[tempa[2]];
                tempa[3] = sbox[tempa[3]];
                
                tempa[0] = tempa[0] ^ Rcon[i / Nk];
            }
            
            j = i * 4;
            k = (i - Nk) * 4;
            roundKeys[j] = roundKeys[k] ^ tempa[0];
            roundKeys[j + 1] = roundKeys[k + 1] ^ tempa[1];
            roundKeys[j + 2] = roundKeys[k + 2] ^ tempa[2];
            roundKeys[j + 3] = roundKeys[k + 3] ^ tempa[3];
        }
    }
    
    private void addRoundKey(int[] state, int round) {
        for (int i = 0; i < 16; i++) {
            state[i] ^= roundKeys[(round * Nb * 4) + i];
        }
    }
    
    private void subBytes(int[] state) {
        for (int i = 0; i < 16; i++) {
            state[i] = sbox[state[i]];
        }
    }
    
    private void invSubBytes(int[] state) {
        for (int i = 0; i < 16; i++) {
            state[i] = invSbox[state[i]];
        }
    }
    
    private void shiftRows(int[] state) {
        int temp;
        
        // Rotate first row 1 columns to left
        temp = state[0 * 4 + 1];
        state[0 * 4 + 1] = state[1 * 4 + 1];
        state[1 * 4 + 1] = state[2 * 4 + 1];
        state[2 * 4 + 1] = state[3 * 4 + 1];
        state[3 * 4 + 1] = temp;
        
        // Rotate second row 2 columns to left
        temp = state[0 * 4 + 2];
        state[0 * 4 + 2] = state[2 * 4 + 2];
        state[2 * 4 + 2] = temp;
        
        temp = state[1 * 4 + 2];
        state[1 * 4 + 2] = state[3 * 4 + 2];
        state[3 * 4 + 2] = temp;
        
        // Rotate third row 3 columns to left
        temp = state[0 * 4 + 3];
        state[0 * 4 + 3] = state[3 * 4 + 3];
        state[3 * 4 + 3] = state[2 * 4 + 3];
        state[2 * 4 + 3] = state[1 * 4 + 3];
        state[1 * 4 + 3] = temp;
    }
    
    private void invShiftRows(int[] state) {
        int temp;
        
        // Rotate first row 1 columns to right
        temp = state[3 * 4 + 1];
        state[3 * 4 + 1] = state[2 * 4 + 1];
        state[2 * 4 + 1] = state[1 * 4 + 1];
        state[1 * 4 + 1] = state[0 * 4 + 1];
        state[0 * 4 + 1] = temp;
        
        // Rotate second row 2 columns to right
        temp = state[0 * 4 + 2];
        state[0 * 4 + 2] = state[2 * 4 + 2];
        state[2 * 4 + 2] = temp;
        
        temp = state[1 * 4 + 2];
        state[1 * 4 + 2] = state[3 * 4 + 2];
        state[3 * 4 + 2] = temp;
        
        // Rotate third row 3 columns to right
        temp = state[0 * 4 + 3];
        state[0 * 4 + 3] = state[1 * 4 + 3];
        state[1 * 4 + 3] = state[2 * 4 + 3];
        state[2 * 4 + 3] = state[3 * 4 + 3];
        state[3 * 4 + 3] = temp;
    }
    
    private int xtime(int x) {
        return ((x << 1) ^ (((x >> 7) & 1) * 0x1b)) & 0xFF;
    }
    
    private void mixColumns(int[] state) {
        int Tmp, Tm, t;
        for (int i = 0; i < 4; ++i) {
            t = state[i * 4 + 0];
            Tmp = state[i * 4 + 0] ^ state[i * 4 + 1] ^ state[i * 4 + 2] ^ state[i * 4 + 3];
            Tm = state[i * 4 + 0] ^ state[i * 4 + 1]; Tm = xtime(Tm); state[i * 4 + 0] ^= Tm ^ Tmp;
            Tm = state[i * 4 + 1] ^ state[i * 4 + 2]; Tm = xtime(Tm); state[i * 4 + 1] ^= Tm ^ Tmp;
            Tm = state[i * 4 + 2] ^ state[i * 4 + 3]; Tm = xtime(Tm); state[i * 4 + 2] ^= Tm ^ Tmp;
            Tm = state[i * 4 + 3] ^ t; Tm = xtime(Tm); state[i * 4 + 3] ^= Tm ^ Tmp;
        }
    }
    
    private int multiply(int x, int y) {
        return (((y & 1) * x) ^
                ((y >> 1 & 1) * xtime(x)) ^
                ((y >> 2 & 1) * xtime(xtime(x))) ^
                ((y >> 3 & 1) * xtime(xtime(xtime(x)))) ^
                ((y >> 4 & 1) * xtime(xtime(xtime(xtime(x)))))) & 0xFF;
    }
    
    private void invMixColumns(int[] state) {
        int a, b, c, d;
        for (int i = 0; i < 4; ++i) {
            a = state[i * 4 + 0];
            b = state[i * 4 + 1];
            c = state[i * 4 + 2];
            d = state[i * 4 + 3];
            
            state[i * 4 + 0] = multiply(a, 0x0e) ^ multiply(b, 0x0b) ^ multiply(c, 0x0d) ^ multiply(d, 0x09);
            state[i * 4 + 1] = multiply(a, 0x09) ^ multiply(b, 0x0e) ^ multiply(c, 0x0b) ^ multiply(d, 0x0d);
            state[i * 4 + 2] = multiply(a, 0x0d) ^ multiply(b, 0x09) ^ multiply(c, 0x0e) ^ multiply(d, 0x0b);
            state[i * 4 + 3] = multiply(a, 0x0b) ^ multiply(b, 0x0d) ^ multiply(c, 0x09) ^ multiply(d, 0x0e);
        }
    }
    
    private void cipher(int[] state) {
        addRoundKey(state, 0);
        
        for (int round = 1; round < Nr; ++round) {
            subBytes(state);
            shiftRows(state);
            mixColumns(state);
            addRoundKey(state, round);
        }
        
        subBytes(state);
        shiftRows(state);
        addRoundKey(state, Nr);
    }
    
    private void invCipher(int[] state) {
        addRoundKey(state, Nr);
        
        for (int round = Nr - 1; round > 0; --round) {
            invShiftRows(state);
            invSubBytes(state);
            addRoundKey(state, round);
            invMixColumns(state);
        }
        
        invShiftRows(state);
        invSubBytes(state);
        addRoundKey(state, 0);
    }
    
    private byte[] pkcs7Pad(byte[] data) {
        int padLen = 16 - (data.length % 16);
        byte[] padded = new byte[data.length + padLen];
        System.arraycopy(data, 0, padded, 0, data.length);
        for (int i = data.length; i < padded.length; i++) {
            padded[i] = (byte) padLen;
        }
        return padded;
    }
    
    private byte[] pkcs7Unpad(byte[] data) {
        if (data.length == 0) return data;
        
        int padLen = data[data.length - 1] & 0xFF;
        if (padLen < 1 || padLen > 16) return data; // Invalid padding
        
        // Verify padding
        for (int i = data.length - padLen; i < data.length; i++) {
            if ((data[i] & 0xFF) != padLen) return data; // Invalid padding
        }
        
        return Arrays.copyOf(data, data.length - padLen);
    }
    
    public byte[] encrypt(byte[] plaintext) {
        byte[] padded = pkcs7Pad(plaintext);
        byte[] encrypted = new byte[padded.length];
        
        for (int i = 0; i < padded.length; i += 16) {
            int[] block = new int[16];
            for (int j = 0; j < 16; j++) {
                block[j] = padded[i + j] & 0xFF;
            }
            cipher(block);
            for (int j = 0; j < 16; j++) {
                encrypted[i + j] = (byte) block[j];
            }
        }
        
        return encrypted;
    }
    
    public byte[] decrypt(byte[] ciphertext) {
        if (ciphertext.length % 16 != 0) {
            throw new IllegalArgumentException("Ciphertext length must be multiple of 16");
        }
        
        byte[] decrypted = new byte[ciphertext.length];
        
        for (int i = 0; i < ciphertext.length; i += 16) {
            int[] block = new int[16];
            for (int j = 0; j < 16; j++) {
                block[j] = ciphertext[i + j] & 0xFF;
            }
            invCipher(block);
            for (int j = 0; j < 16; j++) {
                decrypted[i + j] = (byte) block[j];
            }
        }
        
        return pkcs7Unpad(decrypted);
    }
    
    public String toHex(byte[] data) {
        StringBuilder sb = new StringBuilder();
        for (byte b : data) {
            sb.append(String.format("%02x", b & 0xFF));
        }
        return sb.toString();
    }
    
    public byte[] fromHex(String hex) {
        byte[] result = new byte[hex.length() / 2];
        for (int i = 0; i < result.length; i++) {
            result[i] = (byte) Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
        }
        return result;
    }
    
    // Example usage
    public static void main(String[] args) {
        try {
            // 128-bit key (16 bytes)
            byte[] key = {
                (byte) 0x2b, (byte) 0x7e, (byte) 0x15, (byte) 0x16,
                (byte) 0x28, (byte) 0xae, (byte) 0xd2, (byte) 0xa6,
                (byte) 0xab, (byte) 0xf7, (byte) 0x15, (byte) 0x88,
                (byte) 0x09, (byte) 0xcf, (byte) 0x4f, (byte) 0x3c
            };
            
            AES aes = new AES(key);
            
            // Test with text
            String text = "Hello, AES encryption! This is a test message.";
            byte[] plaintext = text.getBytes(StandardCharsets.UTF_8);
            
            System.out.println("Original text: " + text);
            
            // Encrypt
            byte[] encrypted = aes.encrypt(plaintext);
            System.out.println("Encrypted (hex): " + aes.toHex(encrypted));
            
            // Decrypt
            byte[] decrypted = aes.decrypt(encrypted);
            String decryptedText = new String(decrypted, StandardCharsets.UTF_8);
            System.out.println("Decrypted text: " + decryptedText);
            
            // Test with binary data
            byte[] binaryData = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, (byte) 0xFF, (byte) 0xFE, (byte) 0xFD};
            System.out.println("\nOriginal binary (hex): " + aes.toHex(binaryData));
            
            byte[] encryptedBinary = aes.encrypt(binaryData);
            System.out.println("Encrypted binary (hex): " + aes.toHex(encryptedBinary));
            
            byte[] decryptedBinary = aes.decrypt(encryptedBinary);
            System.out.println("Decrypted binary (hex): " + aes.toHex(decryptedBinary));
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}