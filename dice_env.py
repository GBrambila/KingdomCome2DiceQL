# dice_env.py
import numpy as np
import random
import jogo


def canonical_action(current_roll, move_tuple, decision_bit):
    """
    Dado:
      - current_roll: lista dos dados atuais (na ordem rolada),
      - move_tuple: tupla (ordenada) representando a combinação selecionada (ex.: (5,5)),
      - decision_bit: 0 para "continuar", 1 para "parar".

    Retorna o inteiro canônico (7 bits) que representa essa ação.
    Para isso, escolhe-se os índices mais à esquerda (primeiras ocorrências) para representar o move_tuple.
    """
    chosen_indices = []
    used = [False] * len(current_roll)
    for value in move_tuple:
        for i, die in enumerate(current_roll):
            if not used[i] and die == value:
                used[i] = True
                chosen_indices.append(i)
                break
    canonical_mask = 0
    for i in chosen_indices:
        canonical_mask |= (1 << i)
    return (decision_bit << 6) | canonical_mask


class DiceTurnEnv:
    """
    Ambiente que simula um turno completo do jogo de dados.

    Estado: vetor de 9 dimensões:
      - [0]: turn_score / 50,
      - [1]: player_score / 50,
      - [2]: opponent_score / 50,
      - [3-8]: valores dos 6 dados (com padding se necessário), normalizados por 6.

    Ação: inteiro entre 0 e 127 (7 bits):
      - Bits 0 a 5: máscara que indica quais dados (das posições disponíveis) serão selecionados.
      - Bit 6: decisão — 0 significa “continuar” o turno; 1 significa “parar” e bancar os pontos.

    O método step() aplica a jogada (se válida), atualiza o turno e retorna (próximo estado, recompensa, done, info).
    Se (player_score + turn_score) atingir ou ultrapassar o alvo, o turno é encerrado automaticamente.
    """

    def __init__(self, player_score=0, opponent_score=0, target_score=1500):
        self.player_score = player_score
        self.opponent_score = opponent_score
        self.target_score = target_score
        self.turn_score = 0
        self.done = False
        self.current_roll = []
        self.reset()

    def reset(self):
        self.turn_score = 0
        self.done = False
        self.current_roll = jogo.roll_dice(6)
        return self.get_state()

    def get_state(self):
        state = np.zeros(9, dtype=np.float32)
        state[0] = self.turn_score / 50.0
        state[1] = self.player_score / 50.0
        state[2] = self.opponent_score / 50.0
        dice = self.current_roll + [0] * (6 - len(self.current_roll))
        for i in range(6):
            state[3 + i] = dice[i] / 6.0 if dice[i] > 0 else 0.0
        return state

    def get_valid_actions_mask(self):
        """
        Retorna um vetor (numpy array) de 128 posições onde cada posição é 1 se a ação
        (número entre 0 e 127) corresponder à representação canônica de uma combinação válida,
        considerando ambos os valores de decisão (continuar ou parar).
        """
        mask = np.zeros(128, dtype=np.float32)
        # Obtém as jogadas válidas únicas a partir da rolagem atual
        valid_moves = jogo.get_valid_moves(self.current_roll)  # lista de (move_tuple, score)
        canonical_set = set()
        # Para cada jogada única, gera as duas ações canônicas (para continuar e para parar)
        for move_tuple, score in valid_moves:
            for decision in [0, 1]:
                ca = canonical_action(self.current_roll, move_tuple, decision)
                canonical_set.add(ca)
        # Agora, apenas as ações que forem canônicas serão marcadas como válidas
        for a in range(128):
            if a in canonical_set:
                mask[a] = 1.0
        return mask

    def step(self, action):
        if self.done:
            raise Exception("O turno já terminou. Use reset() para iniciar um novo turno.")
        selection_mask = action & 0x3F
        decision_bit = (action >> 6) & 1
        if selection_mask == 0:
            self.done = True
            return self.get_state(), -50.0, self.done, {"error": "Nenhum dado selecionado"}
        selected_indices = []
        for i in range(6):
            if (selection_mask >> i) & 1:
                if i < len(self.current_roll) and self.current_roll[i] != 0:
                    selected_indices.append(i)
                else:
                    self.done = True
                    return self.get_state(), -50.0, self.done, {"error": "Índice de dado inválido"}
        if len(selected_indices) == 0:
            self.done = True
            return self.get_state(), -50.0, self.done, {"error": "Nenhum dado válido selecionado"}
        selected_dice = [self.current_roll[i] for i in selected_indices]
        score = jogo.score_selection(selected_dice)
        if score is None:
            self.done = True
            return self.get_state(), -50.0, self.done, {"error": "Combinação inválida"}
        self.turn_score += score
        new_roll = [self.current_roll[i] for i in range(len(self.current_roll)) if i not in selected_indices]
        if len(new_roll) == 0:
            new_roll = jogo.roll_dice(6)
        self.current_roll = new_roll
        if self.player_score + self.turn_score >= self.target_score:
            self.done = True
            return self.get_state(), self.turn_score, self.done, {"info": "Alvo atingido"}
        if decision_bit == 1:
            self.done = True
            return self.get_state(), self.turn_score, self.done, {"info": "Turno encerrado pelo agente"}
        else:
            if not jogo.is_scoring_roll(self.current_roll):
                self.turn_score = 0
                self.done = True
                return self.get_state(), 0.0, self.done, {"info": "Bust"}
            self.current_roll = jogo.roll_dice(len(self.current_roll))
            if not jogo.is_scoring_roll(self.current_roll):
                self.turn_score = 0
                self.done = True
                return self.get_state(), 0.0, self.done, {"info": "Bust após nova rolagem"}
            return self.get_state(), 0.0, self.done, {}
