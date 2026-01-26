import AES_1
import math


def pad(data: bytes) -> bytes:
    """
    Pad data to 16-byte boundary with zeros.

    Args:
        data (bytes): Input data to be padded

    Returns:
        bytes: Padded data aligned to 16-byte boundary

    Note:
        If data is already aligned to 16 bytes, returns data unchanged.
    """
    remainder = len(data) % 16
    if remainder == 0:
        return data
    return data + b'\x00' * (16 - remainder)


def xor_bytes(bytes_a: bytes, bytes_b: bytes) -> bytes:
    """
    XOR two byte arrays element-wise.

    Args:
        bytes_a (bytes): First byte array
        bytes_b (bytes): Second byte array

    Returns:
        bytes: Result of XOR operation

    Note:
        Standard operation in stream ciphers. Arrays must be same length.
    """
    return bytes([a ^ b for (a, b) in zip(bytes_a, bytes_b)])


def MUL(x_bytes: bytes, y_bytes: bytes) -> bytes:
    """
    Galois Field multiplication in GF(2^128).

    This performs multiplication in the Galois Field used by GCM mode,
    which is the mathematical foundation for the authentication tag.

    Args:
        x_bytes (bytes): First 16-byte operand
        y_bytes (bytes): Second 16-byte operand

    Returns:
        bytes: 16-byte result of GF(2^128) multiplication

    Note:
        Uses the polynomial 0xe1 << 120 for modular reduction.
        This is the "G" (Galois) in GCM.
    """
    x = int.from_bytes(x_bytes, 'big')
    y = int.from_bytes(y_bytes, 'big')
    r = 0xe1 << 120  # Polynomial used for reduction in GCM

    x_bits = [1 if x & (1 << i) else 0 for i in range(127, -1, -1)]
    z_i = 0
    v_i = y

    for i in range(128):
        if x_bits[i] == 1:
            z_i ^= v_i  # Addition in GF is XOR
        # Right shift and modular reduction
        if (v_i & 1) == 0:
            v_i >>= 1
        else:
            v_i = (v_i >> 1) ^ r
    return z_i.to_bytes(16, 'big')


def GHASH(h: bytes, x: bytes) -> bytes:
    """
    GHASH authentication function.

    Takes data and a hash key (h) and produces a unique cryptographic
    fingerprint used for message authentication in GCM mode.

    Args:
        h (bytes): 16-byte hash key (H) derived from AES encryption of zero block
        x (bytes): Input data to authenticate (must be multiple of 16 bytes)

    Returns:
        bytes: 16-byte authentication hash

    Note:
        Processes data in 16-byte blocks, XORing each block with the previous
        hash result, then multiplying in GF(2^128).
    """
    m = len(x) // 16
    x_blocks = [x[i * 16:i * 16 + 16] for i in range(m)]
    y_i = b'\x00' * 16  # Starting state
    for i in range(m):
        # XOR current block with previous hash, then multiply in GF(2^128)
        y_i = MUL(xor_bytes(y_i, x_blocks[i]), h)
    return y_i


def INC_32(y_bytes: bytes) -> bytes:
    """
    Increment the 32-bit counter part of a 128-bit block.

    GCM mode uses a 128-bit block where only the last 32 bits are
    incremented as a counter, while the first 96 bits remain constant.

    Args:
        y_bytes (bytes): 16-byte block containing counter

    Returns:
        bytes: 16-byte block with incremented counter

    Note:
        Only increments the least significant 32 bits (rightmost 4 bytes).
    """
    y = int.from_bytes(y_bytes, 'big')
    # GCM only increments the last 32 bits of the 128-bit block
    y_inc = ((y >> 32) << 32) ^ (((y & 0xffffffff) + 1) & 0xffffffff)
    return y_inc.to_bytes(16, 'big')


def GCTR(key: bytes, ICB: bytes, x: bytes) -> bytes:
    """
    Galois Counter Mode encryption/decryption function.

    This converts AES (a block cipher) into a stream cipher by encrypting
    counter blocks and XORing them with the plaintext/ciphertext.

    Args:
        key (bytes): 16-byte AES encryption key
        ICB (bytes): 16-byte Initial Counter Block
        x (bytes): Input data to encrypt/decrypt

    Returns:
        bytes: Encrypted/decrypted output (same length as input)

    Note:
        GCTR is symmetric - the same operation encrypts and decrypts.
        Returns empty bytes if input is empty.
    """
    if not x: return b''
    n = math.ceil(len(x) / 16)
    y_blocks = []
    cb_i = ICB

    aes = AES_1.AES(key)
    for i in range(n):
        # Encrypt the Counter Block, then XOR with the plaintext
        keystream_block = aes.encrypt_block(cb_i)

        # Take only the bytes needed if x is not a multiple of 16
        chunk = x[i * 16: i * 16 + 16]
        y_i = xor_bytes(chunk, keystream_block)
        y_blocks.append(y_i)

        # Prepare next counter
        cb_i = INC_32(cb_i)

    return b''.join(y_blocks)


def aes_gcm_encrypt(plaintext: bytes, key: bytes, iv: bytes,associated_data: bytes, tag_length: int) -> tuple[bytes, bytes]:
    """
    Encrypt data using AES-GCM (Galois/Counter Mode).

    AES-GCM provides both confidentiality (encryption) and authenticity
    (authentication tag) in a single operation.

    Args:
        plaintext (bytes): Data to encrypt
        key (bytes): 16-byte AES encryption key
        iv (bytes): Initialization Vector (nonce). Should be 12 bytes for optimal performance
        associated_data (bytes): Additional data to authenticate but not encrypt (AAD)
        tag_length (int): Authentication tag length in bits (typically 128)

    Returns:
        tuple[bytes, bytes]: (ciphertext, authentication_tag)

    Note:
        - IV should be unique for each encryption with the same key
        - AAD is authenticated but not encrypted (e.g., packet headers)
        - Tag must be verified during decryption to ensure authenticity

    Example:
        >>> key = b'sixteen byte key'
        >>> iv = b'twelve bytes'
        >>> ct, tag = aes_gcm_encrypt(b'secret', key, iv, b'', 128)
    """
    aes = AES_1.AES(key)
    h_key = aes.encrypt_block(b'\x00' * 16)  # Hash Key H

    # 1. Generate J0
    if len(iv) == 12:
        j_0 = iv + b'\x00\x00\x00\x01'
    else:
        # Non-96 bit IVs must be GHASHed with padding and length
        iv_padded = pad(iv) + b'\x00' * 8 + (len(iv) * 8).to_bytes(8, 'big')
        j_0 = GHASH(h_key, iv_padded)

    # 2. Encrypt Ciphertext (Starts at J0 + 1)
    ciphertext = GCTR(key, INC_32(j_0), plaintext)

    # 3. Create Auth Input for Tag
    # [AAD + padding] + [Ciphertext + padding] + [64-bit AAD length] + [64-bit CT length]
    auth_input = pad(associated_data) + pad(ciphertext)
    auth_input += (len(associated_data) * 8).to_bytes(8, 'big')
    auth_input += (len(ciphertext) * 8).to_bytes(8, 'big')

    # 4. Final Tag (Uses J0)
    s_hash = GHASH(h_key, auth_input)
    tag = GCTR(key, j_0, s_hash)

    return ciphertext, tag[:tag_length // 8]


def aes_gcm_decrypt(ciphertext: bytes, key: bytes, iv: bytes,associated_data: bytes, tag: bytes, tag_length: int) -> bytes:
    """
    Decrypt data using AES-GCM and verify authentication tag.

    Verifies the authentication tag before returning plaintext. If the tag
    doesn't match, raises an exception to prevent returning corrupted or
    tampered data.

    Args:
        ciphertext (bytes): Encrypted data to decrypt
        key (bytes): 16-byte AES encryption key (same as used for encryption)
        iv (bytes): Initialization Vector (same as used for encryption)
        associated_data (bytes): Additional authenticated data (same as used for encryption)
        tag (bytes): Authentication tag from encryption
        tag_length (int): Tag length in bits (same as used for encryption)

    Returns:
        bytes: Decrypted plaintext

    Raises:
        ValueError: If authentication tag verification fails, indicating
                   tampering or corruption

    Note:
        CRITICAL: Never use decrypted data if tag verification fails.
        The tag verification happens BEFORE decryption is returned.

    Example:
        >>> plaintext = aes_gcm_decrypt(ct, key, iv, b'', tag, 128)
    """
    # Verify tag first, then decrypt
    aes = AES_1.AES(key)
    h_key = aes.encrypt_block(b'\x00' * 16)

    # Generate J0
    if len(iv) == 12:
        j_0 = iv + b'\x00\x00\x00\x01'
    else:
        iv_padded = pad(iv) + b'\x00' * 8 + (len(iv) * 8).to_bytes(8, 'big')
        j_0 = GHASH(h_key, iv_padded)

    # Verify tag
    auth_input = pad(associated_data) + pad(ciphertext)
    auth_input += (len(associated_data) * 8).to_bytes(8, 'big')
    auth_input += (len(ciphertext) * 8).to_bytes(8, 'big')

    s_hash = GHASH(h_key, auth_input)
    expected_tag = GCTR(key, j_0, s_hash)[:tag_length // 8]

    if expected_tag != tag:
        raise ValueError("Authentication failed - tag mismatch")

    # Decrypt
    plaintext = GCTR(key, INC_32(j_0), ciphertext)
    return plaintext


if __name__ == "__main__":
    # Add this test at the end of your main block:
    print("\n=== Testing Encryption/Decryption Round-trip ===")
    test_key = bytearray.fromhex('fe47fcce5fc32665d2ae399e4eec72ba')
    test_iv = bytearray.fromhex('5adb9609dbaeb58cbd6e7275')
    test_plaintext = b"Hello, this is a secret message!"
    test_aad = b"metadata"

    ct, tag = aes_gcm_encrypt(test_plaintext, test_key, test_iv, test_aad, 128)
    recovered_pt = aes_gcm_decrypt(ct, test_key, test_iv, test_aad, tag, 128)

    assert recovered_pt == test_plaintext
    print(f"Original:  {test_plaintext}")
    print(f"Recovered: {recovered_pt}")
    print("✓ Round-trip test passed!")

    # Test tag verification failure
    try:
        bad_tag = bytes([tag[0] ^ 1]) + tag[1:]  # Flip one bit
        aes_gcm_decrypt(ct, test_key, test_iv, test_aad, bad_tag, 128)
        print("✗ Should have failed tag verification!")
    except ValueError as e:
        print(f"✓ Tag verification working: {e}")

    # NIST Special Publication 800-38D

    # NIST test vector 1
    key = bytearray.fromhex('11754cd72aec309bf52f7687212e8957')
    iv = bytearray.fromhex('3c819d9a9bed087615030b65')
    plaintext = bytearray.fromhex('')
    associated_data = bytearray.fromhex('')
    expected_ciphertext = bytearray.fromhex('')
    expected_tag = bytearray.fromhex('250327c674aaf477aef2675748cf6971')
    tag_length = 128

    ciphertext, auth_tag = aes_gcm_encrypt(plaintext, key, iv, associated_data, tag_length)

    assert (ciphertext == expected_ciphertext)
    assert (auth_tag == expected_tag)

    # NIST test vector 2
    key = bytearray.fromhex('fe47fcce5fc32665d2ae399e4eec72ba')
    iv = bytearray.fromhex('5adb9609dbaeb58cbd6e7275')
    plaintext = bytearray.fromhex('7c0e88c88899a779228465074797cd4c2e1498d259b54390b85e3eef1c02df60e743f1b840382c4bccaf'
                                  '3bafb4ca8429bea063')
    associated_data = bytearray.fromhex('88319d6e1d3ffa5f987199166c8a9b56c2aeba5a')
    expected_ciphertext = bytearray.fromhex('98f4826f05a265e6dd2be82db241c0fbbbf9ffb1c173aa83964b7cf5393043736365253ddb'
                                            'c5db8778371495da76d269e5db3e')
    expected_tag = bytearray.fromhex('291ef1982e4defedaa2249f898556b47')
    tag_length = 128

    ciphertext, auth_tag = aes_gcm_encrypt(plaintext, key, iv, associated_data, tag_length)
    assert (ciphertext == expected_ciphertext)
    assert (auth_tag == expected_tag)

    # NIST test vector 3
    key = bytearray.fromhex('c7d9358af0fd737b118dbf4347fd252a')
    iv = bytearray.fromhex('83de9fa52280522b55290ebe3b067286d87690560179554153cb3341a04e15c5f35390602fa07e5b5f16dc38cf0'
                           '82b11ad6dd3fab8552d2bf8d9c8981bbfc5f3b57e5e3066e3df23f078fa25bce63d3d6f86ce9fbc2c679655b958'
                           'b09a991392eb93b453ba6e7bf8242f8f61329e3afe75d0f8536aa7e507d75891e540fb1d7e')
    plaintext = bytearray.fromhex('422f46223fddff25fc7a6a897d20dc8af6cc8a37828c90bd95fa9b943f460eb0a26f29ffc483592efb64'
                                  '835774160a1bb5c0cd')
    associated_data = bytearray.fromhex('5d2b9a4f994ffaa03000149956c8932e85b1a167294514e388b73b10808f509ea73c075ecbf43c'
                                        'ecfec13c202afed62110dabf8026d237f4e765853bc078f3afe081d0a1f8d8f7556b8e42acc3cc'
                                        'e888262185048d67c55b2df1')
    expected_ciphertext = bytearray.fromhex('86eba4911578ac72ac30c25fe424da9ab625f29b5c00e36d2c24a2733dc40123dc57a8c9f1'
                                            '7a24a26c09c73ad4efbcba3bab5b')
    expected_tag = bytearray.fromhex('492305190344618cab8b40f006a57186')
    tag_length = 128

    ciphertext, auth_tag = aes_gcm_encrypt(plaintext, key, iv, associated_data, tag_length)

    assert (ciphertext == expected_ciphertext)
    assert (auth_tag == expected_tag)

    print("\n✓ All NIST test vectors passed!")
