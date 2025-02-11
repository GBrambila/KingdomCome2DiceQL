import random
from collections import defaultdict


# ============================
# FUNÇÕES AUXILIARES DE PONTUAÇÃO
# ============================

def get_counts(roll):
    """
    Dada uma lista de dados, retorna uma tupla com a contagem para cada face (1 a 6).
    """
    counts = [0] * 6
    for die in roll:
        counts[die - 1] += 1
    return tuple(counts)


def is_move_valid(move, counts):
    """
    Verifica se o movimento (um vetor com a quantidade de cada face usada)
    é válido, isto é, não usa mais dados do que os disponíveis.
    """
    return all(m <= c for m, c in zip(move, counts))


def subtract_counts(counts, move):
    """Subtrai, face a face, a quantidade de 'move' de 'counts'."""
    return tuple(c - m for c, m in zip(counts, move))


def add_counts(move1, move2):
    """Soma, face a face, dois movimentos (ou vetores de contagem)."""
    return tuple(a + b for a, b in zip(move1, move2))


def get_elementary_moves(counts):
    """
    A partir da contagem dos dados (tuple), retorna os movimentos elementares pontuáveis.
    Esses movimentos incluem:
      - Runs: 1-6 (1500 pts), 1-5 (500 pts) e 2-6 (750 pts);
      - Trincas (ou mais) para cada face:
            * Trinca de 1: 1000 pts (dados adicionais dobram o valor);
            * Para outras faces: face × 100 (dados extras dobram);
      - Pontuação individual: cada 1 vale 100 e cada 5 vale 50.
    """
    moves = []
    # Run completo 1-6
    if all(c >= 1 for c in counts):
        moves.append((1500, (1, 1, 1, 1, 1, 1)))
    # Run 1-5 (faces 1 a 5)
    if all(counts[i] >= 1 for i in range(5)):
        moves.append((500, (1, 1, 1, 1, 1, 0)))
    # Run 2-6 (faces 2 a 6)
    if all(counts[i] >= 1 for i in range(1, 6)):
        moves.append((750, (0, 1, 1, 1, 1, 1)))
    # Trinca (ou mais) para cada face
    for i in range(6):
        if counts[i] >= 3:
            for k in range(3, counts[i] + 1):
                base = 1000 if i == 0 else (i + 1) * 100
                score_val = base * (2 ** (k - 3))
                move = [0] * 6
                move[i] = k
                moves.append((score_val, tuple(move)))
    # Pontuação individual para 1 (face 1)
    if counts[0] > 0:
        for k in range(1, counts[0] + 1):
            move = [0] * 6
            move[0] = k
            moves.append((100 * k, tuple(move)))
    # Pontuação individual para 5 (face 5)
    if counts[4] > 0:
        for k in range(1, counts[4] + 1):
            move = [0] * 6
            move[4] = k
            moves.append((50 * k, tuple(move)))
    # Filtra apenas movimentos válidos e remove duplicatas (mantendo o de maior pontuação para cada combinação)
    valid_moves = [(score, move) for score, move in moves if is_move_valid(move, counts)]
    unique = {}
    for score, move in valid_moves:
        if move in unique:
            unique[move] = max(unique[move], score)
        else:
            unique[move] = score
    return [(score, move) for move, score in unique.items()]


def enumerate_moves(counts, memo=None):
    """
    Recursivamente, enumera todas as combinações de movimentos possíveis a partir de 'counts'.
    Cada combinação é uma dupla (score_total, move_total).
    Utiliza memoização para otimizar a recursão.
    """
    if memo is None:
        memo = {}
    if counts in memo:
        return memo[counts]
    moves = []
    elementary = get_elementary_moves(counts)
    if not elementary:
        memo[counts] = []
        return []
    for score, move in elementary:
        new_counts = subtract_counts(counts, move)
        sub_moves = enumerate_moves(new_counts, memo)
        # Opção de usar somente este movimento
        moves.append((score, move))
        for sub_score, sub_move in sub_moves:
            combined_score = score + sub_score
            combined_move = add_counts(move, sub_move)
            moves.append((combined_score, combined_move))
    # Remove duplicatas
    unique = {}
    for s, m in moves:
        if m in unique:
            unique[m] = max(unique[m], s)
        else:
            unique[m] = s
    result = [(score, move) for move, score in unique.items()]
    memo[counts] = result
    return result


def select_best_move(roll, bonus=50):
    """
    A partir da rolagem (lista de inteiros), seleciona a melhor combinação de pontuação
    usando a heurística: valor = score + bonus * (dados restantes após a jogada).
    Retorna uma tupla (score, move, novos_dados) ou None se nenhum movimento for possível (bust).
    """
    counts = get_counts(roll)
    moves = enumerate_moves(counts)
    if not moves:
        return None  # bust
    best_move = None
    best_value = -float('inf')
    roll_count = len(roll)
    for score, move in moves:
        used = sum(move)
        # Se usar todos os dados, ganha "hot dice" (reinicia com 6)
        new_dice = roll_count - used if (roll_count - used) > 0 else 6
        value = score + bonus * new_dice
        if value > best_value:
            best_value = value
            best_move = (score, move, new_dice)
    return best_move


# ============================
# AMBIENTE DO JOGO
# ============================

class DiceGameEnv:
    """
    Ambiente do jogo de dados (baseado no Farkle adaptado).

    Estado:
      (my_total_score, turn_score, dice_remaining, opponent_total_score)

    Métodos:
      - reset(): reinicia todo o jogo (usado para treinamento) com um placar de oponente sorteado.
      - new_turn(): inicia um novo turno, resetando apenas a pontuação do turno e os dados disponíveis.
      - step(action): executa a ação (0 -> Rolar, 1 -> Parar)
         * Se rolar: se não houver combinação, ocorre bust (turno finalizado).
         * Se parar: os pontos do turno são somados à pontuação total.
    """

    def __init__(self, goal_score=1500, opponent_total_score=None):
        self.goal_score = goal_score
        self.total_score = 0
        self.turn_score = 0
        self.dice_remaining = 6
        # Se não for especificado, para treinamento sorteia o placar do oponente.
        if opponent_total_score is None:
            self.opponent_total_score = random.randint(0, goal_score)
        else:
            self.opponent_total_score = opponent_total_score
        self.done = False

    def reset(self):
        """Reinicia o jogo inteiro (usado para treinamento)."""
        self.total_score = 0
        self.turn_score = 0
        self.dice_remaining = 6
        self.opponent_total_score = random.randint(0, self.goal_score)
        self.done = False
        return (self.total_score, self.turn_score, self.dice_remaining, self.opponent_total_score)

    def new_turn(self):
        """Inicia um novo turno, resetando apenas o turno e os dados, mantendo os placares."""
        self.turn_score = 0
        self.dice_remaining = 6
        self.done = False

    def step(self, action):
        """
        Executa a ação:
          - action 0: Rolar os dados.
          - action 1: Parar e bancar os pontos do turno.
        Retorna: (novo_estado, recompensa, turn_over, info)
        """
        if self.done:
            return (self.total_score, self.turn_score, self.dice_remaining, self.opponent_total_score), 0, True, {}

        if action == 0:  # Rolar
            roll = [random.randint(1, 6) for _ in range(self.dice_remaining)]
            best_move = select_best_move(roll, bonus=50)
            if best_move is None:
                # Bust: turno finalizado; perde os pontos acumulados no turno.
                reward = -self.turn_score
                self.turn_score = 0
                self.dice_remaining = 6
                self.done = True
                return (
                self.total_score, self.turn_score, self.dice_remaining, self.opponent_total_score), reward, True, {
                    "roll": roll, "bust": True}
            else:
                move_score, move_used, new_dice = best_move
                self.turn_score += move_score
                self.dice_remaining = new_dice
                # "Hot dice": se usar todos os dados, reinicia com 6.
                if self.dice_remaining == 0:
                    self.dice_remaining = 6
                reward = 0  # A recompensa efetiva só ocorre ao parar (ou no bust).
                return (
                self.total_score, self.turn_score, self.dice_remaining, self.opponent_total_score), reward, False, {
                    "roll": roll, "move": best_move}

        elif action == 1:  # Parar
            reward = self.turn_score
            self.total_score += self.turn_score
            self.turn_score = 0
            self.dice_remaining = 6
            self.done = True
            return (self.total_score, self.turn_score, self.dice_remaining, self.opponent_total_score), reward, True, {}
        else:
            raise ValueError("Ação inválida! Use 0 para rolar ou 1 para parar.")


# ============================
# TREINAMENTO COM Q-LEARNING
# ============================

def q_learning_train(episodes=20000, alpha=0.1, gamma=0.9, epsilon=0.1, verbose=False):
    """
    Treina o agente utilizando Q-Learning.
    Cada episódio corresponde a um turno. O estado inclui o placar do oponente,
    possibilitando ao agente aprender estratégias (por exemplo, ser mais conservador se estiver à frente).

    Retorna a Q-table e estatísticas do treinamento.
    """
    env = DiceGameEnv()
    Q = {}  # Tabela Q: chave = (state, action), valor = Q(s, a)
    stats_rewards = []
    stats_steps = []

    for ep in range(episodes):
        state = env.reset()  # Estado: (total_score, turn_score, dice_remaining, opponent_total_score)
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            # Política ε-greedy:
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                q_roll = Q.get((state, 0), 0.0)
                q_stop = Q.get((state, 1), 0.0)
                action = 0 if q_roll >= q_stop else 1

            next_state, reward, done, info = env.step(action)
            # Atualização do Q-Learning:
            current_q = Q.get((state, action), 0.0)
            next_q = max(Q.get((next_state, 0), 0.0), Q.get((next_state, 1), 0.0))
            new_q = current_q + alpha * (reward + gamma * next_q - current_q)
            Q[(state, action)] = new_q

            state = next_state
            ep_reward += reward
            steps += 1

        stats_rewards.append(ep_reward)
        stats_steps.append(steps)
        if verbose and (ep + 1) % 1000 == 0:
            print(f"Episódio {ep + 1}/{episodes}: Recompensa = {ep_reward}, Passos = {steps}")

    training_stats = {'rewards': stats_rewards, 'steps': stats_steps}
    return Q, training_stats


# ============================
# JOGO: HUMANO vs AGENTE (CONTROLADO POR Q-LEARNING)
# ============================

def human_turn(env):
    """
    Executa o turno do jogador humano.
    O estado inclui: (total_score, turn_score, dice_remaining, opponent_total_score).
    Ao final do turno, retorna a pontuação total do jogador.
    """
    env.new_turn()
    while True:
        print(
            f"\nSeu estado atual: Total = {env.total_score}, Turno = {env.turn_score}, Dados restantes = {env.dice_remaining}, Placar do oponente = {env.opponent_total_score}")
        action_input = input("Escolha ação: (r)olar ou (p)arar: ").strip().lower()
        if action_input == 'r':
            action = 0
        elif action_input == 'p':
            action = 1
        else:
            print("Ação inválida. Digite 'r' para rolar ou 'p' para parar.")
            continue

        state, reward, turn_over, info = env.step(action)
        if action == 0:
            print("Você rolou:", info.get("roll"))
            if info.get("bust"):
                print("Bust! Nenhuma combinação pontuável. Você perdeu os pontos deste turno.")
        elif action == 1:
            print("Você parou e bancou os pontos do turno. Pontos ganhos:", reward)

        if turn_over:
            break
    return env.total_score


def agent_turn(Q, env):
    """
    Executa o turno do agente (controlado pela Q-Learning).
    Ao final do turno, retorna a pontuação total do agente.
    """
    env.new_turn()
    while True:
        state = (env.total_score, env.turn_score, env.dice_remaining, env.opponent_total_score)
        q_roll = Q.get((state, 0), 0.0)
        q_stop = Q.get((state, 1), 0.0)
        action = 0 if q_roll >= q_stop else 1
        state, reward, turn_over, info = env.step(action)
        print(f"Agente - Ação: {'Rolar' if action == 0 else 'Parar'} | Estado: {state} | Info: {info}")
        if turn_over:
            break
    return env.total_score


def play_match(Q, goal_score=1500):
    """
    Executa uma partida entre o jogador humano e o agente treinado.
    Cada jogador alterna turnos até que um alcance a meta.
    Os ambientes são atualizados para refletir o placar do oponente.
    """
    # Inicialmente, ambos começam com 0 pontos.
    human_env = DiceGameEnv(goal_score, opponent_total_score=0)
    agent_env = DiceGameEnv(goal_score, opponent_total_score=0)

    while human_env.total_score < goal_score and agent_env.total_score < goal_score:
        # Atualiza os placares dos oponentes antes do turno
        human_env.opponent_total_score = agent_env.total_score
        agent_env.opponent_total_score = human_env.total_score

        print("\n--- Seu turno ---")
        human_turn(human_env)
        print("Sua pontuação total:", human_env.total_score)
        if human_env.total_score >= goal_score:
            break

        # Atualiza o placar do oponente para o agente
        human_env.opponent_total_score = agent_env.total_score
        agent_env.opponent_total_score = human_env.total_score

        print("\n--- Turno do Agente ---")
        agent_turn(Q, agent_env)
        print("Pontuação do Agente:", agent_env.total_score)

    if human_env.total_score >= goal_score:
        print("\nParabéns, você venceu!")
    else:
        print("\nO agente venceu!")


# ============================
# FUNÇÃO PRINCIPAL
# ============================

def main():
    print("Treinando o agente com Q-Learning...")
    Q_table, training_stats = q_learning_train(episodes=10000000, alpha=0.1, gamma=0.9, epsilon=0.1, verbose=True)

    avg_reward = sum(training_stats['rewards']) / len(training_stats['rewards'])
    avg_steps = sum(training_stats['steps']) / len(training_stats['steps'])
    print("\nTreinamento concluído: 20000 episódios")
    print(f"Recompensa média por episódio: {avg_reward:.2f}")
    print(f"Passos médios por episódio: {avg_steps:.2f}\n")

    play_input = input("Deseja jogar contra o agente? (s/n): ").strip().lower()
    if play_input == 's':
        play_match(Q_table, goal_score=1500)
    else:
        print("Fim do programa.")


if __name__ == "__main__":
    main()
