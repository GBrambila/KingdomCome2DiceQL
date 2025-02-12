# controller.py
import numpy as np
import torch
import random
from dice_env import DiceTurnEnv
import jogo
from collections import Counter
from train_ai import DQNAgent

class Player:
    def __init__(self, name, player_type="human"):
        """
        player_type pode ser:
         - "human" para jogador humano,
         - "bot" para estratégia simples,
         - "ai" para agente treinado.
        """
        self.name = name
        self.player_type = player_type
        self.total_score = 0

def decode_action(action, current_roll):
    """
    Decodifica a ação (inteiro 0-127) para extrair:
      - selected_dice: lista dos valores dos dados selecionados (baseado na ordem da rolagem atual);
      - decision_bit: 0 significa continuar; 1 significa parar.
    """
    selection_mask = action & 0x3F
    decision_bit = (action >> 6) & 1
    selected_indices = []
    for i in range(6):
        if (selection_mask >> i) & 1:
            if i < len(current_roll):
                selected_indices.append(i)
    selected_dice = [current_roll[i] for i in selected_indices]
    return selected_dice, decision_bit

def human_turn(player, opponent_score, target_score):
    env = DiceTurnEnv(player_score=player.total_score, opponent_score=opponent_score, target_score=target_score)
    state = env.get_state()
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
        next_state, reward, done, info = env.step(chosen_action)
        if reward != 0:
            if "error" in info:
                print("Erro:", info["error"])
            elif "info" in info:
                print("Info:", info["info"])
            else:
                print("Recompensa obtida:", reward)
        state = next_state
    return env.turn_score

def bot_turn(player, opponent_score, target_score):
    env = DiceTurnEnv(player_score=player.total_score, opponent_score=opponent_score, target_score=target_score)
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
        # Estratégia simples: se já acumulou 300 pontos, tenta escolher uma ação com decisão "Parar"
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
        if env.turn_score >= 50:
            stop_actions = [act for act in valid_actions if ((act >> 6) & 1) == 1]
            chosen_action = stop_actions[0] if stop_actions else candidate
        else:
            chosen_action = candidate
        selected_dice, decision_bit = decode_action(chosen_action, env.current_roll)
        print(f"{player.name} seleciona {selected_dice} com decisão: {'Parar' if decision_bit == 1 else 'Continuar'}, que vale {jogo.score_selection(selected_dice)} pontos.")
        next_state, reward, done, info = env.step(chosen_action)
        if reward != 0:
            if "error" in info:
                print("Erro:", info["error"])
            elif "info" in info:
                print("Info:", info["info"])
            else:
                print("Recompensa obtida:", reward)
        state = next_state
    return env.turn_score

def ai_turn(player, opponent_score, target_score, agent):
    env = DiceTurnEnv(player_score=player.total_score, opponent_score=opponent_score, target_score=target_score)
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
        next_state, reward, done, info = env.step(action)
        if reward != 0:
            if "error" in info:
                print("Erro:", info["error"])
            elif "info" in info:
                print("Info:", info["info"])
            else:
                print("Recompensa obtida:", reward)
        state = next_state
    return env.turn_score

def player_turn(player, opponent_score, target_score, ai_agent=None):
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
    else:
        raise Exception("Tipo de jogador desconhecido.")

def play_game(player1, player2, target_score):
    """
    Executa um jogo completo entre player1 e player2 até que um deles alcance o target_score.
    Retorna o nome do vencedor.
    """
    players = [player1, player2]
    current_player_index = 0
    ai_agent = None
    if player1.player_type == "ai" or player2.player_type == "ai":
        ai_agent = DQNAgent(input_dim=9, output_dim=128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ai_agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=device))
        ai_agent.policy_net.eval()
    while True:
        current_player = players[current_player_index]
        opponent = players[1 - current_player_index]
        print("\n------------------------------------------")
        print(f"É a vez de {current_player.name} (Pontuação total: {current_player.total_score})")
        turn_points = player_turn(current_player, opponent.total_score, target_score, ai_agent)
        current_player.total_score += turn_points
        print(f"{current_player.name} encerrou o turno com {turn_points} pontos.")
        print(f"Pontuação total de {current_player.name}: {current_player.total_score}")
        if current_player.total_score >= target_score:
            print("\n==========================================")
            print(f"{current_player.name} venceu o jogo com {current_player.total_score} pontos!")
            print("==========================================")
            return current_player.name  # Retorna o vencedor
        current_player_index = 1 - current_player_index

class GameController:
    def __init__(self, player1, player2, target_score=1500):
        self.player1 = player1
        self.player2 = player2
        self.target_score = target_score

    def start(self):
        return play_game(self.player1, self.player2, self.target_score)
