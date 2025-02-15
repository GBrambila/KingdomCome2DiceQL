# global_env.py
import numpy as np
import jogo
from dice_env import DiceTurnEnv
import random
# Mapeamento de ação para threshold com 6 ações:
ACTION_TO_THRESHOLD = { 0: 200, 1: 300, 2: 500, 3: 750,4: 1000,5:1250,6:1500}

# Adicionar no final do arquivo (antes da classe GlobalMatchEnv)
def decode_action(action, current_roll):
    """
    Decodifica a ação (inteiro 0-127) para extrair:
      - selected_dice: lista dos valores dos dados selecionados
      - decision_bit: 0 (continuar) ou 1 (parar)
    """
    selection_mask = action & 0x3F  # bits 0-5
    decision_bit = (action >> 6) & 1
    selected_indices = [i for i in range(6) if (selection_mask >> i) & 1]
    selected_dice = [current_roll[i] for i in selected_indices if i < len(current_roll)]
    return selected_dice, decision_bit
def simulate_turn(threshold, player_score, opponent_score, target_score):
    """
    Simula um turno utilizando uma política fixa baseada no limiar de risco.
    Enquanto o turno não acabar, a política é:
      - Se o turno acumulado for menor que o threshold, escolher uma ação que continue (decision bit = 0).
      - Se o turno acumulado for maior ou igual ao threshold, escolher uma ação que pare (decision bit = 1), se disponível.
    Caso contrário, seleciona a primeira ação válida.
    Retorna a pontuação acumulada no turno.
    """
    env = DiceTurnEnv(player_score=player_score, opponent_score=opponent_score, target_score=target_score)
    while not env.done:
        valid_mask = env.get_valid_actions_mask()
        valid_actions = [a for a in range(128) if valid_mask[a] == 1]
        if not valid_actions:
            env.done = True
            break
        continue_actions = [a for a in valid_actions if ((a >> 6) & 1) == 0]
        stop_actions = [a for a in valid_actions if ((a >> 6) & 1) == 1]
        if env.turn_score >= threshold and stop_actions:
            # Nova lógica para escolher a melhor ação de parar
            best_score = -1
            best_action = None
            for action in stop_actions:
                selected_dice, _ = decode_action(action, env.current_roll)
                score = jogo.score_selection(selected_dice)
                if score is not None and score > best_score:
                    best_score = score
                    best_action = action
            if best_action is not None:
                chosen_action = best_action
            else:
                chosen_action = stop_actions[0]  # Fallback seguro
        elif env.turn_score < threshold and continue_actions:
            chosen_action = continue_actions[0]
        else:
            chosen_action = valid_actions[0]
        _, _, done, info = env.step(chosen_action)
    return env.turn_score


class GlobalMatchEnv:
    """
    Ambiente para simular uma partida completa usando uma política parametrizada.

    Estado: vetor de dimensão 2: [score_current/target, score_opponent/target]
    Ação: inteiro em {0,1,2,3,4,5} que determina o threshold, conforme ACTION_TO_THRESHOLD.

    O ambiente alterna entre os jogadores e termina quando um deles alcança o target_score.
    A recompensa final é definida como: (score do jogador atual / 50) e, se ele vencer, soma +100.
    """

    def __init__(self, target_score=1500):
        self.target_score = target_score
        self.reset()

    def reset(self):
        self.scores = [0, 0]  # [player0, player1]
        self.current_player = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        cp = self.current_player
        op = 1 - cp
        return np.array([self.scores[cp] / self.target_score, self.scores[op] / self.target_score], dtype=np.float32)

    def step(self, action):
        """
        Ação: inteiro em {0,1,2,3,4,5} mapeado para um threshold.
        Simula o turno do jogador atual usando o threshold escolhido.
        Atualiza o placar e alterna o turno.

        Retorna: next_state, reward, done, info

        Quando o episódio termina, o reward é calculado como:
            reward = (score do jogador atual / 50)
            Se o jogador atual venceu (score >= target), então reward += 100.
        """
        threshold = ACTION_TO_THRESHOLD.get(action, 300)
        cp = self.current_player
        op = 1 - cp
        turn_score = simulate_turn(threshold, self.scores[cp], self.scores[op], self.target_score)
        self.scores[cp] += turn_score
        done = False
        info = {}
        # Verifica se a partida terminou:
        if self.scores[cp] >= self.target_score or self.scores[op] >= self.target_score:
            done = True
            # A recompensa é a pontuação final do jogador corrente dividida por 50...
            reward = self.scores[cp] / 50.0
            # ... e, se ele venceu (atingiu ou ultrapassou o target), adiciona +100.
            if self.scores[cp] >= self.target_score:
                reward = 100
            info["final_scores"] = self.scores.copy()
        else:
            reward = 0.0
        self.current_player = 1 - self.current_player
        self.done = done
        next_state = self._get_state()
        return next_state, reward, done, info
