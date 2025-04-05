# Dingo Optimization Algorithm

import time
import math
import numpy as np
from Enc_Dec import key_generation, chaCha20_encrypt


def initialize_dingoes(num_dingoes, dim, bounds):
    """Initialize the position of dingoes within the given bounds."""
    return bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(num_dingoes, dim)


def objective_function(x):
    encrypted_data, private_key = key_generation(x)

    # encryption

    df_scaled = np.array(x)
    enc_start = time.time()  # 256-bit key for ChaCha20
    cipher_text, nonce = chaCha20_encrypt(df_scaled, private_key)
    enc_end = time.time()
    encryption_time = enc_end - enc_start

    return encryption_time, private_key


def dingo_optimization(data, num_dingoes, dim, bounds, max_iter):
    """Dingo Optimization Algorithm (DOX)"""
    # Initialize dingoes
    dingoes = initialize_dingoes(num_dingoes, dim, bounds)
    fitness, optimal_key = objective_function(data)

    # Identify alpha (best), beta (second best), and delta (third best) dingoes
    sorted_indices = np.argsort(fitness)
    alpha, beta, delta = dingoes[sorted_indices][0][:3]

    # Algorithm parameters
    a = 3  # Linearly decreasing parameter

    for iteration in range(max_iter):
        a = a - (iteration * (3 / max_iter))  # Decrease linearly from 3 to 0

        for i in range(num_dingoes):
            r1, r2 = np.random.rand(), np.random.rand()
            A1, C1 = 2 * a * r1 - a, 2 * r2  # Coefficients

            D_alpha = np.abs(C1 * alpha - dingoes[i])
            X1 = alpha - A1 * D_alpha

            r3, r4 = np.random.rand(), np.random.rand()
            A2, C2 = 2 * a * r3 - a, 2 * r4

            D_beta = np.abs(C2 * beta - dingoes[i])
            X2 = beta - A2 * D_beta

            r5, r6 = np.random.rand(), np.random.rand()
            A3, C3 = 2 * a * r5 - a, 2 * r6

            D_delta = np.abs(C3 * delta - dingoes[i])
            X3 = delta - A3 * D_delta

            dingoes[i] = (X1 + X2 + X3) / 3

        # Evaluate new positions
        fitness, optimal_key = objective_function(data)

        # Update alpha, beta, and delta
        alpha, beta, delta = dingoes[sorted_indices][0][:3]

    return alpha, fitness, optimal_key



