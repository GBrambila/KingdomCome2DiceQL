# jogo.py
import random
from collections import Counter
import itertools


def roll_dice(n):
    """Rola n dados e retorna uma lista com os resultados."""
    return [random.randint(1, 6) for _ in range(n)]


def compute_score_for_counter(counter, memo):
    """
    Função recursiva que, dado um multiconjunto de dados (representado por um Counter),
    retorna a pontuação máxima possível se for possível particionar TODOS os dados em
    combinações válidas; caso contrário, retorna None.

    Combinações válidas:
      - Dados individuais: 1 vale 100 pontos; 5 vale 50.
      - Trinca (ou mais) de um mesmo número:
           • Trinca de 1 vale 1000; de outros números vale (n × 100).
           • Cada dado extra além da trinca dobra a pontuação da combinação.
      - Sequências:
           • 1-2-3-4-5 => 500 pontos.
           • 2-3-4-5-6 => 750 pontos.
           • 1-2-3-4-5-6 => 1500 pontos.
    """
    key = tuple(counter[i] for i in range(1, 7))
    if key in memo:
        return memo[key]
    if sum(counter.values()) == 0:
        return 0  # Todos os dados foram usados

    best = None

    # Dados individuais para 1 e 5
    for face in [1, 5]:
        if counter[face] > 0:
            new_counter = counter.copy()
            new_counter[face] -= 1
            score_val = 100 if face == 1 else 50
            sub = compute_score_for_counter(new_counter, memo)
            if sub is not None:
                candidate = score_val + sub
                if best is None or candidate > best:
                    best = candidate

    # Trincas (ou mais) – cada dado adicional dobra a pontuação da trinca
    for face in range(1, 7):
        if counter[face] >= 3:
            for group_size in range(3, counter[face] + 1):
                new_counter = counter.copy()
                new_counter[face] -= group_size
                base = 1000 if face == 1 else face * 100
                score_val = base * (2 ** (group_size - 3))
                sub = compute_score_for_counter(new_counter, memo)
                if sub is not None:
                    candidate = score_val + sub
                    if best is None or candidate > best:
                        best = candidate

    # Sequências
    # Sequência 1-2-3-4-5 => 500 pontos
    if all(counter[i] >= 1 for i in [1, 2, 3, 4, 5]):
        new_counter = counter.copy()
        for i in [1, 2, 3, 4, 5]:
            new_counter[i] -= 1
        sub = compute_score_for_counter(new_counter, memo)
        if sub is not None:
            candidate = 500 + sub
            if best is None or candidate > best:
                best = candidate

    # Sequência 2-3-4-5-6 => 750 pontos
    if all(counter[i] >= 1 for i in [2, 3, 4, 5, 6]):
        new_counter = counter.copy()
        for i in [2, 3, 4, 5, 6]:
            new_counter[i] -= 1
        sub = compute_score_for_counter(new_counter, memo)
        if sub is not None:
            candidate = 750 + sub
            if best is None or candidate > best:
                best = candidate

    # Sequência completa 1-2-3-4-5-6 => 1500 pontos
    if all(counter[i] >= 1 for i in [1, 2, 3, 4, 5, 6]):
        new_counter = counter.copy()
        for i in [1, 2, 3, 4, 5, 6]:
            new_counter[i] -= 1
        sub = compute_score_for_counter(new_counter, memo)
        if sub is not None:
            candidate = 1500 + sub
            if best is None or candidate > best:
                best = candidate

    memo[key] = best
    return best


def score_selection(selected_dice):
    """
    Dada uma seleção de dados (lista de inteiros), retorna a pontuação se a seleção
    for válida (isto é, se for possível particioná-la integralmente em combinações pontuáveis).
    Retorna a pontuação ou None se a seleção for inválida.
    """
    memo = {}
    counter = Counter(selected_dice)
    score = compute_score_for_counter(counter, memo)
    return score


def get_valid_moves(roll):
    """
    A partir de uma rolagem (lista de inteiros), gera todas as jogadas válidas.
    Cada jogada é representada como uma tupla (seleção, pontuação), onde a seleção
    é uma tupla ordenada dos dados escolhidos.

    Como o número de dados é pequeno (máximo 6), a iteração sobre os subconjuntos é viável.
    """
    valid_moves = {}
    n = len(roll)
    for r in range(1, n + 1):
        for indices in itertools.combinations(range(n), r):
            selection = [roll[i] for i in indices]
            selection_sorted = tuple(sorted(selection))
            if selection_sorted in valid_moves:
                continue
            score_val = score_selection(selection)
            if score_val is not None:
                valid_moves[selection_sorted] = score_val
    return list(valid_moves.items())


def is_scoring_roll(roll):
    """Retorna True se a rolagem contém pelo menos uma jogada válida."""
    return len(get_valid_moves(roll)) > 0
