# main.py
import controller

def main():
    # ==============================
    # Painel de Configuração Inicial
    # ==============================
    # Defina o modo de jogo desejado (como string):
    # "1": Humano vs Bot
    # "2": Bot vs Bot
    # "3": Humano vs Humano
    # "4": Humano vs IA (turn-based)
    # "5": IA (turn-based) vs Bot
    # "6": IA (turn-based) vs IA (turn-based)
    # "7": Humano vs Global AI
    # "8": Global AI vs Bot
    # "9": Global AI vs Global AI
    mode = "8" 

    # Defina a pontuação alvo (ex.: 1500 ou 2000)
    target_score = 1500

    # Defina a quantidade de partidas a serem jogadas
    num_games = 500  # Por exemplo, 3 partidas

    # Configuração dos jogadores (definidos no painel)
    if mode == "1":
        player1_name = "Alice"
        player1_type = "human"
        player2_name = "Bot"
        player2_type = "bot"
    elif mode == "2":
        player1_name = "Bot 1"
        player1_type = "bot"
        player2_name = "Bot 2"
        player2_type = "bot"
    elif mode == "3":
        player1_name = "Alice"
        player1_type = "human"
        player2_name = "Bob"
        player2_type = "human"
    elif mode == "4":
        player1_name = "Alice"
        player1_type = "human"
        player2_name = "IA"
        player2_type = "ai"
    elif mode == "5":
        player1_name = "IA"
        player1_type = "ai"
        player2_name = "Bot"
        player2_type = "bot"
    elif mode == "6":
        player1_name = "IA 1"
        player1_type = "ai"
        player2_name = "IA 2"
        player2_type = "ai"
    elif mode == "7":
        player1_name = "Alice"
        player1_type = "human"
        player2_name = "Global AI"
        player2_type = "global_ai"
    elif mode == "8":
        player1_name = "Global AI"
        player1_type = "global_ai"
        player2_name = "Bot"
        player2_type = "bot"
    elif mode == "9":
        player1_name = "Global AI"
        player1_type = "global_ai"
        player2_name = "IA"
        player2_type = "ai"
    elif mode == "10":
        player1_name = "Global AI 1"
        player1_type = "global_ai"
        player2_name = "Global AI 2"
        player2_type = "global_ai"
    else:
        print("Modo de jogo inválido. Encerrando.")
        return

    scoreboard = {player1_name: 0, player2_name: 0}

    for game_num in range(1, num_games + 1):
        print(f"\n\n=== Iniciando Partida {game_num} de {num_games} ===")
        player1 = controller.Player(player1_name, player_type=player1_type)
        player2 = controller.Player(player2_name, player_type=player2_type)
        game_controller = controller.GameController(player1, player2, target_score)
        winner = game_controller.start()
        scoreboard[winner] += 1

    print("\n\n=== Placar Final ===")
    for name, wins in scoreboard.items():
        print(f"{name}: {wins} vitória(s)")

if __name__ == "__main__":
    main()
