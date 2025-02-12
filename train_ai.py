# train_ai.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from dice_env import DiceTurnEnv
import os


# --- Definição da rede neural DQN ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_valid_mask):
        self.buffer.append((state, action, reward, next_state, done, next_valid_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, next_valid_mask = map(np.array, zip(*batch))
        return state, action, reward, next_state, done, next_valid_mask

    def __len__(self):
        return len(self.buffer)


# --- Agente DQN com Action Masking ---
class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.1, epsilon_decay=5000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state, valid_mask):
        self.steps_done += 1
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        if random.random() < epsilon:
            valid_indices = np.where(valid_mask == 1)[0]
            if len(valid_indices) == 0:
                return random.randint(0, 127)
            return int(np.random.choice(valid_indices))
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
            q_values[valid_mask == 0] = -np.inf
            return int(np.argmax(q_values))

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones, next_valid_masks = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        next_valid_masks = torch.FloatTensor(next_valid_masks).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        next_q_values = self.target_net(next_states)
        next_q_values[next_valid_masks == 0] = -1e9
        max_next_q_values, _ = next_q_values.max(1)
        max_next_q_values = max_next_q_values.unsqueeze(1)

        expected_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# --- Loop de Treinamento ---
def train(num_episodes=1000, batch_size=64, target_update=1000, buffer_capacity=10000):
    env = DiceTurnEnv()  # ambiente com pontuação inicial zero
    input_dim = 9
    output_dim = 128
    agent = DQNAgent(input_dim, output_dim)
    replay_buffer = ReplayBuffer(buffer_capacity)

    episode_rewards = []
    losses = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        while True:
            valid_mask = env.get_valid_actions_mask()
            action = agent.select_action(state, valid_mask)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            next_valid_mask = env.get_valid_actions_mask() if not done else np.zeros(128, dtype=np.float32)
            replay_buffer.push(state, action, reward, next_state, done, next_valid_mask)
            state = next_state
            loss = agent.update(replay_buffer, batch_size)
            if loss is not None:
                losses.append(loss)
            if done:
                break
        episode_rewards.append(total_reward)

        if episode % target_update == 0:
            agent.update_target()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episódio {episode + 1}, Recompensa Média: {avg_reward:.2f}, Loss Média: {avg_loss:.4f}")

    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("Treinamento concluído. Modelo salvo em dqn_model.pth")


if __name__ == "__main__":
    train()
