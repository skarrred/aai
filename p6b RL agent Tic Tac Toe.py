#p6b
#B) Teach an RL agent to play Tic-Tac-Toe against itself or a user using rewards for wins, losses and draws.Reward: +1 for win, -1 for loss, 0 for draw
import random

Q = {}
alpha = 0.5
gamma = 0.9

def init():
    return [' '] * 9

def moves(s):
    return [i for i in range(9) if s[i] == ' ']

def win(s, p):
    win_positions = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    return any(s[a] == s[b] == s[c] == p for a,b,c in win_positions)

def draw(s):
    return ' ' not in s

def best_move(s, p):
    state_key = str(s)
    if state_key not in Q:
        Q[state_key] = [0]*9
    possible_moves = moves(s)
    if not possible_moves:
        return None
    q_values = [Q[state_key][i] for i in possible_moves]
    max_q = max(q_values)
    max_indices = [possible_moves[i] for i, q in enumerate(q_values) if q == max_q]
    return random.choice(max_indices)

def play(episodes=5000):
    for _ in range(episodes):
        s = init()
        history = []
        while True:
            for p in ['X', 'O']:
                a = best_move(s, p)
                if a is None:
                    break
                s[a] = p
                history.append((str(s), a, p))
                if win(s, p):
                    r = 1 if p == 'X' else -1
                    break
                if draw(s):
                    r = 0
                    break
            else:
                continue
            break

        # Q-learning update backward through history
        for state_key, action, player in reversed(history):
            if state_key not in Q:
                Q[state_key] = [0]*9
            Q[state_key][action] += alpha * (r - Q[state_key][action])
            r = -r  # Flip reward for the other player

play()
print("Training complete!")
    