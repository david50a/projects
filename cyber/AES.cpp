#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

class AES {
private:
    static const int Nb = 4;
    static const int Nk = 4;
    static const int Nr = 10;

    // S-box
    static const unsigned char sbox[256];
    static const unsigned char inv_sbox[256];

    // Round constants
    static const unsigned char Rcon[11];

    unsigned char roundKeys[176]; // 4 * Nb * (Nr + 1)

    inline unsigned char getSBoxValue(unsigned char num) const { return sbox[num]; }
    inline unsigned char getSBoxInvert(unsigned char num) const { return inv_sbox[num]; }

    void keyExpansion(const unsigned char* key) {
        int i, j, k;
        unsigned char tempa[4];

        // Copy the key into the first round key (first 16 bytes)
        for (i = 0; i < Nk; i++) {
            roundKeys[i*4]     = key[i*4];
            roundKeys[i*4 + 1] = key[i*4 + 1];
            roundKeys[i*4 + 2] = key[i*4 + 2];
            roundKeys[i*4 + 3] = key[i*4 + 3];
        }

        for (i = Nk; i < Nb * (Nr + 1); i++) {
            k = (i - 1) * 4;
            tempa[0] = roundKeys[k];
            tempa[1] = roundKeys[k + 1];
            tempa[2] = roundKeys[k + 2];
            tempa[3] = roundKeys[k + 3];

            if (i % Nk == 0) {
                // Rotate word
                unsigned char u8tmp = tempa[0];
                tempa[0] = tempa[1];
                tempa[1] = tempa[2];
                tempa[2] = tempa[3];
                tempa[3] = u8tmp;

                // SubBytes
                tempa[0] = getSBoxValue(tempa[0]);
                tempa[1] = getSBoxValue(tempa[1]);
                tempa[2] = getSBoxValue(tempa[2]);
                tempa[3] = getSBoxValue(tempa[3]);

                tempa[0] = tempa[0] ^ Rcon[i / Nk];
            }

            j = i * 4; k = (i - Nk) * 4;
            roundKeys[j]     = roundKeys[k]     ^ tempa[0];
            roundKeys[j + 1] = roundKeys[k + 1] ^ tempa[1];
            roundKeys[j + 2] = roundKeys[k + 2] ^ tempa[2];
            roundKeys[j + 3] = roundKeys[k + 3] ^ tempa[3];
        }
    }

    void addRoundKey(unsigned char* state, int round) {
        // each round uses 16 bytes
        int start = round * Nb * 4;
        for (int i = 0; i < 16; i++) {
            state[i] ^= roundKeys[start + i];
        }
    }

    void subBytes(unsigned char* state) {
        for (int i = 0; i < 16; i++) state[i] = getSBoxValue(state[i]);
    }

    void invSubBytes(unsigned char* state) {
        for (int i = 0; i < 16; i++) state[i] = getSBoxInvert(state[i]);
    }

    // State is column-major: state[row + 4*col]
    void shiftRows(unsigned char* state) {
        unsigned char temp[16];
        // copy state
        for (int i = 0; i < 16; ++i) temp[i] = state[i];

        // Row 0 unchanged
        // Row r is shifted left by r
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                state[r + 4*c] = temp[r + 4*((c + r) % 4)];
            }
        }
    }

    void invShiftRows(unsigned char* state) {
        unsigned char temp[16];
        for (int i = 0; i < 16; ++i) temp[i] = state[i];

        // Row r shifted right by r
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                state[r + 4*c] = temp[r + 4*((c - r + 4) % 4)];
            }
        }
    }

    inline unsigned char xtime(unsigned char x) {
        return static_cast<unsigned char>((x << 1) ^ (((x >> 7) & 1) * 0x1b));
    }

    void mixColumns(unsigned char* state) {
        unsigned char Tmp, Tm, t;
        for (int i = 0; i < 4; ++i) {
            t = state[i*4 + 0];
            Tmp = state[i*4 + 0] ^ state[i*4 + 1] ^ state[i*4 + 2] ^ state[i*4 + 3];
            Tm = state[i*4 + 0] ^ state[i*4 + 1]; Tm = xtime(Tm); state[i*4 + 0] ^= Tm ^ Tmp;
            Tm = state[i*4 + 1] ^ state[i*4 + 2]; Tm = xtime(Tm); state[i*4 + 1] ^= Tm ^ Tmp;
            Tm = state[i*4 + 2] ^ state[i*4 + 3]; Tm = xtime(Tm); state[i*4 + 2] ^= Tm ^ Tmp;
            Tm = state[i*4 + 3] ^ t;              Tm = xtime(Tm); state[i*4 + 3] ^= Tm ^ Tmp;
        }
    }

    unsigned char multiply(unsigned char x, unsigned char y) {
        unsigned char result = 0;
        unsigned char a = x;
        unsigned char b = y;
        for (int i = 0; i < 8; ++i) {
            if (b & 1) result ^= a;
            unsigned char hiBitSet = (a & 0x80);
            a <<= 1;
            if (hiBitSet) a ^= 0x1b;
            b >>= 1;
        }
        return result;
    }

    void invMixColumns(unsigned char* state) {
        unsigned char a, b, c, d;
        for (int i = 0; i < 4; ++i) {
            a = state[i*4 + 0];
            b = state[i*4 + 1];
            c = state[i*4 + 2];
            d = state[i*4 + 3];

            state[i*4 + 0] = multiply(a, 0x0e) ^ multiply(b, 0x0b) ^ multiply(c, 0x0d) ^ multiply(d, 0x09);
            state[i*4 + 1] = multiply(a, 0x09) ^ multiply(b, 0x0e) ^ multiply(c, 0x0b) ^ multiply(d, 0x0d);
            state[i*4 + 2] = multiply(a, 0x0d) ^ multiply(b, 0x09) ^ multiply(c, 0x0e) ^ multiply(d, 0x0b);
            state[i*4 + 3] = multiply(a, 0x0b) ^ multiply(b, 0x0d) ^ multiply(c, 0x09) ^ multiply(d, 0x0e);
        }
    }

    void cipher(unsigned char* state) {
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

    void invCipher(unsigned char* state) {
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

    std::vector<unsigned char> pkcs7Pad(const std::vector<unsigned char>& data) {
        int padLen = 16 - (data.size() % 16);
        if (padLen == 0) padLen = 16;
        std::vector<unsigned char> padded = data;
        for (int i = 0; i < padLen; ++i) padded.push_back(static_cast<unsigned char>(padLen));
        return padded;
    }

    std::vector<unsigned char> pkcs7Unpad(const std::vector<unsigned char>& data) {
        if (data.empty()) return data;
        int padLen = static_cast<int>(data.back());
        if (padLen <= 0 || padLen > 16) throw std::runtime_error("Invalid PKCS#7 padding length");
        if (data.size() < static_cast<size_t>(padLen)) throw std::runtime_error("Invalid PKCS#7 padding (data too short)");

        // Verify padding bytes
        for (size_t i = data.size() - padLen; i < data.size(); ++i) {
            if (data[i] != static_cast<unsigned char>(padLen)) throw std::runtime_error("Invalid PKCS#7 padding bytes");
        }
        return std::vector<unsigned char>(data.begin(), data.end() - padLen);
    }

public:
    AES(const std::vector<unsigned char>& key) {
        if (key.size() != 16) throw std::invalid_argument("Key must be 16 bytes for AES-128");
        keyExpansion(key.data());
    }

    std::vector<unsigned char> encrypt(const std::vector<unsigned char>& plaintext) {
        std::vector<unsigned char> padded = pkcs7Pad(plaintext);
        std::vector<unsigned char> encrypted;
        encrypted.reserve(padded.size());
        for (size_t i = 0; i < padded.size(); i += 16) {
            unsigned char block[16];
            std::memcpy(block, &padded[i], 16);
            cipher(block);
            encrypted.insert(encrypted.end(), block, block + 16);
        }
        return encrypted;
    }

    std::vector<unsigned char> decrypt(const std::vector<unsigned char>& ciphertext) {
        if (ciphertext.size() % 16 != 0) throw std::invalid_argument("Ciphertext length must be multiple of 16");
        std::vector<unsigned char> decrypted;
        decrypted.reserve(ciphertext.size());
        for (size_t i = 0; i < ciphertext.size(); i += 16) {
            unsigned char block[16];
            std::memcpy(block, &ciphertext[i], 16);
            invCipher(block);
            decrypted.insert(decrypted.end(), block, block + 16);
        }
        return pkcs7Unpad(decrypted);
    }

    std::string toHex(const std::vector<unsigned char>& data) const {
        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (unsigned char byte : data) {
            ss << std::setw(2) << (static_cast<unsigned int>(byte) & 0xff);
        }
        return ss.str();
    }

    std::vector<unsigned char> fromHex(const std::string& hex) const {
        if (hex.size() % 2) throw std::invalid_argument("Hex string must have even length");
        std::vector<unsigned char> out;
        out.reserve(hex.size() / 2);
        for (size_t i = 0; i < hex.size(); i += 2) {
            std::string byteStr = hex.substr(i, 2);
            unsigned int val = static_cast<unsigned int>(std::stoul(byteStr, nullptr, 16));
            out.push_back(static_cast<unsigned char>(val & 0xff));
        }
        return out;
    }
};

// S-box
const unsigned char AES::sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

// Inverse S-box
const unsigned char AES::inv_sbox[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
};

// Rcon: note index 1 = 0x01 ...
const unsigned char AES::Rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

// Example usage/test
int main() {
    try {
        std::vector<unsigned char> key = {
            0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,
            0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c
        };

        AES aes(key);

        std::string text = "Hello, AES encryption! This is a test message.";
        std::vector<unsigned char> plaintext(text.begin(), text.end());

        std::cout << "Original text: " << text << "\n";

        auto encrypted = aes.encrypt(plaintext);
        std::cout << "Encrypted (hex): " << aes.toHex(encrypted) << "\n";

        auto decrypted = aes.decrypt(encrypted);
        std::string decryptedText(decrypted.begin(), decrypted.end());
        std::cout << "Decrypted text: " << decryptedText << "\n";

        // binary test
        std::vector<unsigned char> binaryData = {0x00,0x01,0x02,0x03,0x04,0x05,0xFF,0xFE,0xFD};
        std::cout << "\nOriginal binary (hex): " << aes.toHex(binaryData) << "\n";
        auto encBin = aes.encrypt(binaryData);
        std::cout << "Encrypted binary (hex): " << aes.toHex(encBin) << "\n";
        auto decBin = aes.decrypt(encBin);
        std::cout << "Decrypted binary (hex): " << aes.toHex(decBin) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
