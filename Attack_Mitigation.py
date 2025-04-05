import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
from save_load import load

class DDoSMitigationAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])  # Exploit

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), np.array([target_f[0]]), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define Actions
ACTIONS = {
    0: "Allow Traffic",
    1: "Apply Rate Limiting",
    2: "Block Suspicious IP",
    3: "Redirect Traffic"
}

# Simulated predicted results from attack detection model (0 = Normal, 1 = Attack)
predicted_results = load('predicted_80')

# Initialize RL Agent
agent = DDoSMitigationAgent(state_size=1, action_size=4)

# Process Each Traffic Instance
for i, state in enumerate(predicted_results):
    action = agent.act([state])  # Get RL-based action
    action_message = ACTIONS[action]

    if state == 1:
        print(f"🚨 [ALERT] Attack Detected at Traffic Instance {i+1}! Applying Mitigation: {action_message}")
    else:
        print(f"✅ [SAFE] Normal Traffic at Instance {i+1}. Decision: Allow Traffic")

    # Simulate environment feedback and training
    reward = 10 if action in [1, 2] else -10  # Reward system (better for blocking/rate limiting)
    next_state = 0  # Assume state resets after mitigation
    done = True  # One step per instance

    agent.remember([state], action, reward, [next_state], done)
    agent.train(batch_size=32)
