import numpy as np
import torch
import random
from dice_env import DiceTurnEnv
import jogo
from collections import Counter
from train_ai import DQNAgent  # para IA local
from train_global_ai import GlobalDQNAgent  # para IA global

# ==============================
# Definição da classe Player
# ==============================
class Player:
    def __init__(self, name, player_type="human"):
        """
        player_type: "human", "bot", "ai" (local) ou "global_ai" (global match).
        """
        self.name = name
        self.player_type = player_type
        self.total_score = 0

# ==============================
# Função para turno do jogador humano
# ==============================
def human_turn(player, opponent_score, target_score):
    env = DiceTurnEnv(player_score=player.total_score,
                      opponent_score=opponent_score,
                      target_score=target_score)
    while not env.done:
        print(f"\n{player.name} - Pontos acumulados neste turno: {env.turn_score}")
        print(f"Rolando {len(env.current_roll)} dados...")
        print("Resultado da rolagem:", env.current_roll)
        valid_mask = env.get_valid_actions_mask()
        valid_actions = [action for action in range(128) if valid_mask[action] == 1]
        if not valid_actions:
            print("BUST! Nenhuma jogada válida. Turno perdido.")
            env.done = True
            break
        print("Jogadas válidas disponíveis:")
        for idx, act in enumerate(valid_actions):
            selected_dice, decision_bit = decode_action(act, env.current_roll)
            score = jogo.score_selection(selected_dice)
            decision_str = "Parar" if decision_bit == 1 else "Continuar"
            print(f"{idx+1}: {selected_dice} => {score} pontos, decisão: {decision_str}")
        while True:
            try:
                choice = int(input("Escolha uma jogada pelo número: "))
                if 1 <= choice <= len(valid_actions):
                    chosen_action = valid_actions[choice-1]
                    break
                else:
                    print("Opção inválida.")
            except:
                print("Entrada inválida.")
        _, reward, done, info = env.step(chosen_action)
        if reward != 0:
            if "error" in info:
                print("Erro:", info["error"])
            elif "info" in info:
                print("Info:", info["info"])
            else:
                print("Recompensa obtida:", reward)
    return env.turn_score

# ==============================
# Função para turno do bot (estratégia simples)
# ==============================
def bot_turn(player, opponent_score, target_score):
    env = DiceTurnEnv(player_score=player.total_score,
                      opponent_score=opponent_score,
                      target_score=target_score)
    while not env.done:
        print(f"\n{player.name} - Pontos acumulados neste turno: {env.turn_score}")
        print(f"Rolando {len(env.current_roll)} dados...")
        print("Resultado da rolagem:", env.current_roll)
        valid_mask = env.get_valid_actions_mask()
        valid_actions = [action for action in range(128) if valid_mask[action] == 1]
        if not valid_actions:
            print("BUST! Nenhuma jogada válida. Turno perdido.")
            env.done = True
            break
        # Estratégia simples: se acumulou 300 pontos, tenta parar
        candidate = None
        best_dice_count = -1
        best_score = -1
        for act in valid_actions:
            selected_dice, decision_bit = decode_action(act, env.current_roll)
            score = jogo.score_selection(selected_dice)
            if score is None:
                continue
            dice_count = len(selected_dice)
            if dice_count > best_dice_count or (dice_count == best_dice_count and score > best_score):
                best_dice_count = dice_count
                best_score = score
                candidate = act
        if env.turn_score >= 300:
            stop_actions = [act for act in valid_actions if ((act >> 6) & 1) == 1]
            chosen_action = stop_actions[0] if stop_actions else candidate
        else:
            chosen_action = candidate
        selected_dice, decision_bit = decode_action(chosen_action, env.current_roll)
        decision_str = "Parar" if decision_bit == 1 else "Continuar"
        print(f"{player.name} seleciona {selected_dice} com decisão: {decision_str}, que vale {jogo.score_selection(selected_dice)} pontos.")
        _, reward, done, info = env.step(chosen_action)
        if reward != 0:
            if "error" in info:
                print("Erro:", info["error"])
            elif "info" in info:
                print("Info:", info["info"])
            else:
                print("Recompensa obtida:", reward)
    return env.turn_score

# ==============================
# Função para turno da IA local
# ==============================
def ai_turn(player, opponent_score, target_score, agent):
    env = DiceTurnEnv(player_score=player.total_score,
                      opponent_score=opponent_score,
                      target_score=target_score)
    device = agent.device
    while not env.done:
        print(f"\n{player.name} (IA) - Pontos acumulados neste turno: {env.turn_score}")
        print(f"Rolando {len(env.current_roll)} dados...")
        print("Resultado da rolagem:", env.current_roll)
        valid_mask = env.get_valid_actions_mask()
        state = env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor).cpu().numpy().flatten()
        q_values[valid_mask == 0] = -np.inf
        action = int(np.argmax(q_values))
        selected_dice, decision_bit = decode_action(action, env.current_roll)
        score = jogo.score_selection(selected_dice)
        decision_str = "Parar" if decision_bit == 1 else "Continuar"
        print(f"{player.name} (IA) seleciona {selected_dice} => {score} pontos, decisão: {decision_str}")
        _, reward, done, info = env.step(action)
        if reward != 0:
            if "error" in info:
                print("Erro:", info["error"])
            elif "info" in info:
                print("Info:", info["info"])
            else:
                print("Recompensa obtida:", reward)
    return env.turn_score

# ==============================
# Função para turno da Global IA
# ==============================
def global_ai_turn(player, opponent_score, target_score, global_agent):
    """
    Simula um turno utilizando a política global.
    O estado do ambiente global é definido como [player_score/target, opponent_score/target].
    O agente global escolhe uma ação (um inteiro de 0 a 5) que determina o threshold.
    Em seguida, o turno é simulado utilizando a função simulate_turn() do global_env.
    São exibidos os detalhes básicos do turno.
    """
    from global_env import simulate_turn  # importa a função definida em global_env.py
    print(f"\n{player.name} (Global AI) - Pontos acumulados neste turno: (inicialmente 0)")
    state = np.array([player.total_score / target_score, opponent_score / target_score], dtype=np.float32)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(global_agent.device)
    with torch.no_grad():
        q_values = global_agent.policy_net(state_tensor).cpu().numpy().flatten()
    action = int(np.argmax(q_values))
    threshold = {0:50, 1:100, 2:200, 3:300, 4:500, 5:750}.get(action, 300)
    print(f"{player.name} (Global AI) escolheu threshold = {threshold} para o turno.")
    from global_env import simulate_turn
    turn_points = simulate_turn(threshold, player.total_score, opponent_score, target_score)
    print(f"{player.name} (Global AI) obteve {turn_points} pontos neste turno.")
    return turn_points


# ==============================
# Função para decodificar a ação
# ==============================
def decode_action(action, current_roll):
    """
    Decodifica a ação (inteiro 0-127) para extrair:
      - selected_dice: lista dos valores dos dados selecionados (baseado na ordem da rolagem atual);
      - decision_bit: 0 significa continuar; 1 significa parar.
    """
    selection_mask = action & 0x3F  # bits 0-5
    decision_bit = (action >> 6) & 1
    selected_indices = []
    for i in range(6):
        if (selection_mask >> i) & 1:
            if i < len(current_roll):
                selected_indices.append(i)
    selected_dice = [current_roll[i] for i in selected_indices]
    return selected_dice, decision_bit

# ==============================
# Função que escolhe o turno do jogador de acordo com seu tipo
# ==============================
def player_turn(player, opponent_score, target_score, ai_agent=None, global_agent=None):
    if player.player_type == "human":
        return human_turn(player, opponent_score, target_score)
    elif player.player_type == "bot":
        return bot_turn(player, opponent_score, target_score)
    elif player.player_type == "ai":
        if ai_agent is None:
            ai_agent = DQNAgent(input_dim=9, output_dim=128)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ai_agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=device))
            ai_agent.policy_net.eval()
        return ai_turn(player, opponent_score, target_score, ai_agent)
    elif player.player_type == "global_ai":
        if global_agent is None:
            global_agent = GlobalDQNAgent(input_dim=2, output_dim=7)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            global_agent.policy_net.load_state_dict(torch.load("global_dqn_model.pth", map_location=device))
            global_agent.policy_net.eval()
        return global_ai_turn(player, opponent_score, target_score, global_agent)
    else:
        raise Exception("Tipo de jogador desconhecido.")

# ==============================
# Função que gerencia o jogo completo
# ==============================
def play_game(player1, player2, target_score):
    players = [player1, player2]
    current_player_index = 0
    ai_agent = None
    global_agent = None
    if player1.player_type in ["ai"] or player2.player_type in ["ai"]:
        ai_agent = DQNAgent(input_dim=9, output_dim=128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ai_agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=device))
        ai_agent.policy_net.eval()
    if player1.player_type in ["global_ai"] or player2.player_type in ["global_ai"]:
        global_agent = GlobalDQNAgent(input_dim=2, output_dim=7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_agent.policy_net.load_state_dict(torch.load("global_dqn_model.pth", map_location=device))
        global_agent.policy_net.eval()
    while True:
        current_player = players[current_player_index]
        opponent = players[1 - current_player_index]
        print("\n------------------------------------------")
        print(f"É a vez de {current_player.name} (Pontuação total: {current_player.total_score})")
        turn_points = player_turn(current_player, opponent.total_score, target_score, ai_agent, global_agent)
        current_player.total_score += turn_points
        print(f"{current_player.name} encerrou o turno com {turn_points} pontos.")
        print(f"Pontuação total de {current_player.name}: {current_player.total_score}")
        if current_player.total_score >= target_score:
            print("\n==========================================")
            print(f"{current_player.name} venceu o jogo com {current_player.total_score} pontos!")
            print("==========================================")
            return current_player.name
        current_player_index = 1 - current_player_index

# ==============================
# Classe GameController
# ==============================
class GameController:
    def __init__(self, player1, player2, target_score=1500):
        self.player1 = player1
        self.player2 = player2
        self.target_score = target_score

    def start(self):
        return play_game(self.player1, self.player2, self.target_score)
