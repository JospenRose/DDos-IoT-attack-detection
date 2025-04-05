from phe import paillier  # Homomorphic Encryption
from Crypto.Cipher import ChaCha20


def key_generation(data):
    public_key, private_key = paillier.generate_paillier_keypair()
    encrypted_data = [public_key.encrypt(x) for x in data]
    private_key = str(private_key.p_inverse)[:32]
    private_key = private_key.encode()
    return encrypted_data, private_key


def chaCha20_encrypt(data, key):
    cipher = ChaCha20.new(key=key)
    data = data.tobytes()
    ciphertext = cipher.encrypt(data)
    return ciphertext, cipher.nonce
