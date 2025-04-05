import pandas as pd
import numpy as np
import json
import requests
from sklearn.model_selection import train_test_split
from save_load import save
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from Dingo import dingo_optimization
from Crypto.Cipher import ChaCha20
from Enc_Dec import chaCha20_encrypt
import base64
import seaborn as sns
import matplotlib.pyplot as plt


def datagen():
    data = pd.read_csv('Dataset/dataset_sdn.csv')

    label = data['label']

    label = np.array(label)

    data = data.drop(columns=['src', 'dst', 'label', 'Protocol'])

    # Correlation Heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.xticks(fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.title("Correlation Heatmap", fontweight='bold', fontname='Serif')
    plt.savefig("Data visualization/Correlation Heatmap.png")
    plt.show()

    # Preprocessing
    # Handling missing values - KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Min-Max Normalization
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=data.columns)

    # Optimal Key generation - Dingo Optimizer
    # dim :  Dimensionality of the problem
    # bounds :  Lower and upper bounds
    # num_dingoes : Population size
    # max_iter : Maximum iterations

    best_solution, best_fitness, optimal_key = dingo_optimization(df_scaled.iloc[0].tolist(), num_dingoes=30, dim=5, bounds=(-10, 10), max_iter=2)

    df_scaled = np.array(df_scaled)
    cipher_text, nonce = chaCha20_encrypt(df_scaled, optimal_key)
    print("ChaCha20 Cipher Text:", cipher_text)

    # Convert encrypted data into a string for storage
    encrypted_data_str = json.dumps({
        'cipher_text': base64.b64encode(cipher_text).decode('utf-8'),
        'nonce': base64.b64encode(nonce).decode('utf-8')
    })
    response = requests.post('http://127.0.0.1:5000/mine_block', json={'encrypted_data': encrypted_data_str})

    print(response.json())

    response = requests.get("http://127.0.0.1:5000/get_chain")
    blockchain_data = response.json()
    encrypted_data_str = blockchain_data['chain'][-1]['encrypted_data']
    encrypted_data = json.loads(encrypted_data_str)

    cipher_text = base64.b64decode(encrypted_data['cipher_text'])
    nonce = base64.b64decode(encrypted_data['nonce'])

    decipher = ChaCha20.new(key=optimal_key, nonce=nonce)
    decrypted_data = decipher.decrypt(cipher_text)

    decrypted_data = np.frombuffer(decrypted_data, dtype=np.float64)
    feat_data = decrypted_data.reshape(data.shape[0], data.shape[1])

    if df_scaled.all() == feat_data.all():
        print("Data Successfully Decrypted....")

    learning_rates = [0.7, 0.8]

    for train_size in learning_rates:
        x_train, x_test, y_train, y_test = train_test_split(feat_data, label, train_size=train_size)
        save('x_train_' + str(int(train_size*100)), x_train)
        save('y_train_' + str(int(train_size * 100)), y_train)
        save('x_test_' + str(int(train_size * 100)), x_test)
        save('y_test_' + str(int(train_size * 100)), y_test)

