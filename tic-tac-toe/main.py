import random

# This function checks if there is a winner and if yes then who

def winner(b):
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

    for a,c,d in lines:
        if b[a] != 0 and b[a] == b[c] == b[d]:
            return b[a]
    return 0 

# checks if the board is completely filled

def is_full(b):
    return 0 not in b 



def available_moves(b):
    return [i for i, v in enumerate(b) if v == 0]


# all reachable states

V = {}


def init_states(b, player):

    if b in V:
        return

    w = winner(b)

    if w == 1:
        V[b] = 1.0 # agent won
    elif w == -1 or is_full(b):
        V[b] = 0.0 # agent lost
    else:
        V[b] = 0.5 
        for m in available_moves(b):
            nb = list(b); nb[m] = player
            init_states(tuple(nb), -player)

init_states((0,) *9, 1)

def agent_move(b, eps=0.1):
    moves = available_moves(b)
    if random.random() < eps:
        return random.choice(moves)

    best, best_v = None, -1

    for m in moves:
        nb = list(b); nb[m] = 1
        v = V[tuple(nb)]

        if v > best_v:
            best_v, best = v, m 


    return best

def opponent_move(b):
    return random.choice(available_moves(b))

def train( episodes = 20000, alpha = 0.1 , eps = 0.1):
    for _ in range(episodes):
        b = (0,)*9
        prev_state = None 

        while True:
            m = agent_move(b, eps)
            nb = list(b); nb[m] = 1; b = tuple(nb)


            if prev_state is not None:
                V[prev_state] += alpha * (V[b] - V[prev_state])
            prev_state = b

            if winner(b) or is_full(b):
                break

            m = opponent_move(b)
            nb = list(b); nb[m] = -1; b = tuple(nb)

            if winner(b) or is_full(b):
                V[prev_state] += alpha * (V[b] - V[prev_state])
                break



train()

def play_test_game():
    b = (0,)*9
    while True:
        m = agent_move(b, eps=0)  # pure greedy now
        nb = list(b); nb[m] = 1; b = tuple(nb)
        if winner(b) or is_full(b):
            return winner(b)
        m = opponent_move(b)
        nb = list(b); nb[m] = -1; b = tuple(nb)
        if winner(b) or is_full(b):
            return winner(b)

losses = sum(1 for _ in range(1000) if play_test_game() == -1)
print(f"losses out of 1000 test games: {losses}")