import datetime
import hashlib
import json
from flask import Flask, jsonify, request
from Crypto.Cipher import ChaCha20
from phe import paillier  # Homomorphic encryption

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0', encrypted_data="Genesis Block")

    def create_block(self, proof, previous_hash, encrypted_data):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.datetime.now()),
            'proof': proof,
            'previous_hash': previous_hash,
            'encrypted_data': encrypted_data  # Store encrypted data in block
        }
        self.chain.append(block)
        return block

    def print_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False

        while check_proof is False:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:5] == '00000':
                check_proof = True
            else:
                new_proof += 1

        return new_proof

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1

        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False

            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()

            if hash_operation[:5] != '00000':
                return False
            previous_block = block
            block_index += 1

        return True


# Initialize Blockchain
app = Flask(__name__)
blockchain = Blockchain()

@app.route('/mine_block', methods=['POST'])
def mine_block():
    print('--'*15)
    print(request.json)
    print('---'*10)
    data = request.json.get("encrypted_data")  # Expecting encrypted data from the request
    print(data)
    if not data:
        return jsonify({"message": "No data provided!"}), 400

    previous_block = blockchain.print_previous_block()
    proof = blockchain.proof_of_work(previous_block['proof'])
    previous_hash = blockchain.hash(previous_block)

    block = blockchain.create_block(proof, previous_hash, encrypted_data=data)

    response = {
        'message': 'A block is MINED',
        'index': block['index'],
        'timestamp': block['timestamp'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
        'encrypted_data': block['encrypted_data']
    }

    return jsonify(response), 200

@app.route('/get_chain', methods=['GET'])
def display_chain():
    response = {'chain': blockchain.chain, 'length': len(blockchain.chain)}
    return jsonify(response), 200

@app.route('/valid', methods=['GET'])
def valid():
    valid = blockchain.chain_valid(blockchain.chain)
    return jsonify({'message': 'Blockchain is valid' if valid else 'Blockchain is NOT valid'}), 200

app.run(host='127.0.0.1', port=5000)


