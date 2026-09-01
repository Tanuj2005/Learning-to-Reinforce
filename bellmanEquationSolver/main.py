import numpy as np

N = 5
A, A_PRIME, A_R = (0,1), (4,1), 10
B, B_PRIME, B_R = (0,3), (2,3), 5
gamma = 0.9
actions = [(-1,0),(1,0),(0,-1),(0,1)]

def idx(r,c): return r*N+c 

def step(r, c, a):
    if (r,c) == A: return A_PRIME, A_R 
    if (r,c) == B: return B_PRIME, B_R 
    nr, nc = r+a[0], c+a[1]
    if 0 <= nr < N and 0 <= nc < N:
        return (nr, nc), 0
    return (r,c), -1

def solve_vpi( policy_probs ):
    Pmat = np.zeros((N*N, N*N))
    Rvec = np.zeros(N*N)
    for r in range(N):
        for c in range(N):
            s = idx(r,c)
            for ai, p in policy_probs(r,c):
                (nr, nc), rew = step(r,c,actions[ai])
                Pmat[s, idx(nr,nc)] += p 
                Rvec[s] += p * rew 

    v = np.linalg.solve( np.eye(N*N) - gamma*Pmat, Rvec)

    return v.reshape(N, N)



random_policy = lambda r,c: [(ai, 0.25) for ai in range(4)]
v_random = solve_vpi(random_policy)
print("v_pi (random policy):\n", np.round(v_random, 1))

def solve_vstar( tol = 1e-6 ):
    v = np.zeros((N,N))
    while True:
        v_new = np.zeros((N,N))
        for r in range(N):
            for c in range(N):
                qs = []
                for a in actions:
                    (nr, nc), rew = step(r, c, a)
                    qs.append( rew + gamma*v[nr, nc])
                v_new[r,c] = max(qs)
        if np.abs(v_new - v).max() < tol:
            v = v_new; break
        v = v_new
    
    return v


v_star = solve_vstar()
print("v* (optimal):\n", np.round(v_star, 1))

def optimal_policy(v):
    arrows = {0: '^', 1:'v', 2:'<', 3:'>'}

    pi = [['' for _ in range(N)] for _ in range(N)]

    for r in range(N):
        for c in range(N):
            qs = []
            for a in actions:
                (nr, nc), rew = step(r, c, a)
                qs.append(rew + gamma*v[nr, nc])
            best = max(qs)
            pi[r][c] = ''.join(arrows[ai] for ai, q in enumerate(qs) if abs(q-best) < 1e-3)

    return pi 

for row in optimal_policy(v_star):
    print(row)
