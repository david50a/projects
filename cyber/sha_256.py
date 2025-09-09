H_0_words = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def H_0():
    return H_0_words.copy()


K = bytearray.fromhex('428a2f98 71374491 b5c0fbcf e9b5dba5 3956c25b 59f111f1 923f82a4 ab1c5ed5'
                      'd807aa98 12835b01 243185be 550c7dc3 72be5d74 80deb1fe 9bdc06a7 c19bf174'
                      'e49b69c1 efbe4786 0fc19dc6 240ca1cc 2de92c6f 4a7484aa 5cb0a9dc 76f988da'
                      '983e5152 a831c66d b00327c8 bf597fc7 c6e00bf3 d5a79147 06ca6351 14292967'
                      '27b70a85 2e1b2138 4d2c6dfc 53380d13 650a7354 766a0abb 81c2c92e 92722c85'
                      'a2bfe8a1 a81a664b c24b8b70 c76c51a3 d192e819 d6990624 f40e3585 106aa070'
                      '19a4c116 1e376c08 2748774c 34b0bcb5 391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3'
                      '748f82ee 78a5636f 84c87814 8cc70208 90befffa a4506ceb bef9a3f7 c67178f2'.replace(" ", ""))


def int_from_bytes(x: bytes) -> int:
    return int.from_bytes(x, 'big')


K_256 = [int_from_bytes(K[i*4:(i+1)*4]) for i in range(len(K) // 4)]


def sha_256_pad_and_parse_message(message: bytes) -> [int]:

    # Padding the Message
    l = len(message) * 8
    n_zeros = (512 - (l + 64 + 1)) % 512

    message += b'\x80'
    message += b'\x00' * ((n_zeros-7) // 8)
    message += l.to_bytes(8, 'big')
    assert ((len(message) * 8) % 512 == 0)

    # Parsing the Message
    block_byte_length = 512 // 8
    N = len(message) // block_byte_length
    word_count = block_byte_length // 4

    message_blocks = []
    for i in range(N):
        block_start = i*block_byte_length
        message_block = [message[block_start + j*4:block_start + (j+1)*4] for j in range(word_count)]
        message_blocks.append([int_from_bytes(mb) for mb in message_block])

    return message_blocks


def shr(x: int, n: int) -> int:
    return x >> n


def rotr(x: int, n: int) -> int:
    return (x >> n) | (x << (32-n)) & 0xffffffff


def Sig_0(x: int) -> int:
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def Sig_1(x: int) -> int:
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def sig_0(x: int) -> int:
    return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)


def sig_1(x: int) -> int:
    return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)


def Ch(x, y, z):
    return (x & y) ^ (~x & z)


def Maj(x, y, z):
    return (x & y) ^ (x & z) ^ (y & z)


def sha_256(message: bytes) -> bytes:

    # Preprocessing
    H = H_0()
    message_blocks = sha_256_pad_and_parse_message(message)

    # Hash computation
    N = len(message_blocks)
    for i in range(1, N+1):

        # Prepare the message schedule
        W = [message_blocks[i-1][t] for t in range(16)]

        for t in range(16, 64):
            W.append((sig_1(W[t-2]) + W[t-7] + sig_0(W[t-15]) + W[t-16]) & 0xffffffff)

        # Initialize working variables
        a, b, c, d, e, f, g, h = H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7]

        # Shuffle
        for t in range(64):
            T_1 = (h + Sig_1(e) + Ch(e, f, g) + K_256[t] + W[t]) & 0xffffffff
            T_2 = (Sig_0(a) + Maj(a, b, c)) & 0xffffffff
            h = g
            g = f
            f = e
            e = (d + T_1) & 0xffffffff
            d = c
            c = b
            b = a
            a = (T_1 + T_2) & 0xffffffff

        # New intermediate hash values
        H[0] += a
        H[1] += b
        H[2] += c
        H[3] += d
        H[4] += e
        H[5] += f
        H[6] += g
        H[7] += h
        H = [h & 0xffffffff for h in H]

    return b''.join([h.to_bytes(4, 'big') for h in H])