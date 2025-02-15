# train_global_parallel.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from global_env import GlobalMatchEnv
import os


#######################################
# Vectorized Environment for Global AI
#######################################
class VecGlobalMatchEnv:
    def __init__(self, num_envs, target_score=1500):
        self.envs = [GlobalMatchEnv(target_score=target_score) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        states = []
        for env in self.envs:
            states.append(env.reset())
        return np.stack(states)  # shape: (num_envs, state_dim)

    def step(self, actions):
        """
        Executa o step para cada ambiente, dado um array de ações.
        Retorna:
          - next_states: array com o próximo estado para cada ambiente;
          - rewards: array com as recompensas;
          - dones: array com os flags de término;
          - infos: lista de dicionários com informações adicionais.
        """
        next_states = []
        rewards = []
        dones = []
        infos = []
        for env, action in zip(self.envs, actions):
            # Se o episódio já terminou, apenas retorna o estado atual, zero reward e done=True.
            if not hasattr(env, "done") or not env.done:
                ns, r, d, info = env.step(action)
            else:
                ns, r, d, info = env._get_state(), 0.0, True, {}
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return np.stack(next_states), np.array(rewards), np.array(dones), infos


#######################################
# Modelo Global DQN (com 6 saídas)
#######################################
class GlobalDQN(nn.Module):
    def __init__(self, input_dim=2, output_dim=7):
        super(GlobalDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


#######################################
# Replay Buffer
#######################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


#######################################
# Agente Global DQN com suporte a batches
#######################################
class GlobalDQNAgent:
    def __init__(self, input_dim=2, output_dim=7, lr=1e-4, gamma=0.997,
                 epsilon_start=1.0, epsilon_final=0.1, epsilon_decay=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.policy_net = GlobalDQN(input_dim, output_dim).to(self.device)
        self.target_net = GlobalDQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action_batch(self, states):
        batch_size = states.shape[0]
        self.steps_done += batch_size
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        states_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(states_tensor)
        actions = []
        for i in range(batch_size):
            if random.random() < epsilon:
                actions.append(random.randint(0, 5))
            else:
                actions.append(int(q_values[i].argmax()))
        return np.array(actions)

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


#######################################
# Treinamento paralelo da Global IA
#######################################
def train_global_parallel(num_episodes=160000, batch_size=64, num_envs=16, target_update=500, buffer_capacity=10000):
    vec_env = VecGlobalMatchEnv(num_envs, target_score=1500)
    agent = GlobalDQNAgent(input_dim=2, output_dim=7)
    replay_buffer = ReplayBuffer(buffer_capacity)

    episode_rewards = []  # Lista com a recompensa final de cada episódio.
    episode_steps = []  # Lista com o número de passos de cada episódio.
    final_scores_list = []  # Lista com os escores finais (informação extra)

    env_episode_rewards = np.zeros(num_envs)
    env_steps = np.zeros(num_envs)
    total_episodes = 0

    states = vec_env.reset()  # shape: (num_envs, 2)

    while total_episodes < num_episodes:
        actions = agent.select_action_batch(states)  # array de shape (num_envs,)
        next_states, rewards, dones, infos = vec_env.step(actions)

        env_episode_rewards += rewards
        env_steps += 1

        for i in range(num_envs):
            replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

        states = next_states

        for i in range(num_envs):
            if dones[i]:
                episode_rewards.append(env_episode_rewards[i])
                episode_steps.append(env_steps[i])
                # Se houver informações de escores finais, as armazena.
                if "final_scores" in infos[i]:
                    final_scores_list.append(infos[i]["final_scores"])
                total_episodes += 1
                env_episode_rewards[i] = 0.0
                env_steps[i] = 0
                states[i] = vec_env.envs[i].reset()

        loss = agent.update(replay_buffer, batch_size)
        if total_episodes % target_update < num_envs:
            agent.update_target()

        if total_episodes > 0 and total_episodes % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_steps = np.mean(episode_steps[-100:]) if len(episode_steps) >= 100 else np.mean(episode_steps)
            # Se houver escores finais, calcula a média dos escores para o jogador 0 e o jogador 1
            if final_scores_list:
                scores_arr = np.array(final_scores_list)
                avg_score_player0 = np.mean(scores_arr[:, 0])
                avg_score_player1 = np.mean(scores_arr[:, 1])
            else:
                avg_score_player0 = avg_score_player1 = 0
            print(f"Episódios {total_episodes}, Recompensa Média: {avg_reward:.2f}, Passos Médios: {avg_steps:.1f}")
            print(f"Média dos escores finais: Jogador0: {avg_score_player0:.1f}, Jogador1: {avg_score_player1:.1f}")

    torch.save(agent.policy_net.state_dict(), "global_dqn_model.pth")
    print("Treinamento Global em paralelo concluído. Modelo salvo em global_dqn_model.pth")


if __name__ == "__main__":
    train_global_parallel()
