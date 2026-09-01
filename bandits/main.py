import numpy as np
import matplotlib.pyplot as plt

K, RUNS, STEPS = 10, 2000, 1000
rng = np.random.default_rng(0)

q_star = rng.normal( 0, 1, (RUNS, K))
best_action = q_star.argmax(1)

def sample_reward( actions ):
    return rng.normal( q_star[np.arange(RUNS), actions], 1)

def run( steps, select_action, update, Q0 = 0.0 ):

    Q = np.full((RUNS, K), Q0, dtype = float)
    N = np.zeros((RUNS, K))
    R = np.zeros((RUNS, steps))

    OPT = np.zeros((RUNS, steps))

    for t in range(1, steps + 1):
        a = select_action(Q,N, t)
        r = sample_reward(a)

        update(Q, N, a, r, t)

        R[:, t-1] = r
        OPT[:, t-1] = a == best_action

    return R.mean(0), OPT.mean(0) * 100


def eps_greedy( epsilon, alpha = None, Q0 = 0.0):
    def select( Q, N, t):
        explore = rng.random(RUNS) < epsilon
        return np.where(explore, rng.integers(0, K, RUNS), Q.argmax(1))
    def update( Q, N, a, r, t):
        N[np.arange(RUNS), a] += 1
        step = alpha if alpha else 1 / N[np.arange(RUNS), a]

        Q[np.arange(RUNS), a] += step * (r - Q[np.arange(RUNS), a])

    return run(STEPS, select, update, Q0)

def ucb(c):
    def select(Q, N, t):
        bonus = c * np.sqrt(np.log(t) / np.maximum(N, 1e-5))
        return (Q + bonus).argmax(1)

    def update(Q, N, a, r, t):
        N[np.arange(RUNS), a] += 1
        Q[np.arange(RUNS), a] += (r - Q[np.arange(RUNS), a]) / N[np.arange(RUNS), a]

    return run(STEPS, select, update)

def gradient_bandit( alpha, baseline = True):
    avg_r = np.zeros(RUNS)
    def pi(H):
        e = np.exp(H - H.max(1, keepdims = True))
        return e / e.sum(1, keepdims = True)

    def select( H, N, t):
        p = pi(H)
        return (p.cumsum(1) > rng.random((RUNS, 1))).argmax(1)

    def update(H, N, a, r, t):
        nonlocal avg_r

        p = pi(H)
        b = avg_r if baseline else 0
        one_hot = np.eye(K)[a]
        H += alpha * ( r - b )[:, None] * (one_hot - p)
        avg_r = avg_r + (r - avg_r) / t 

    return run( STEPS, select, update)


def plot(fname, title, series):
    """series: list of (label, avg_reward_curve, pct_optimal_curve)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    for label, r, o in series:
        a1.plot(r, label=label); a2.plot(o, label=label)
    a1.set_ylabel("Average reward"); a1.set_title(title); a1.legend()
    a2.set_ylabel("% Optimal action"); a2.set_xlabel("Steps"); a2.legend()
    plt.tight_layout(); plt.savefig(fname); plt.close()
 
# Figure 2.2 -- epsilon-greedy comparison
r0, o0 = eps_greedy(0.0)
r1, o1 = eps_greedy(0.1)
r2, o2 = eps_greedy(0.01)
plot("fig2_2.png", "Fig 2.2 -- epsilon-greedy",
     [("eps=0 (greedy)", r0, o0), ("eps=0.1", r1, o1), ("eps=0.01", r2, o2)])
 
# Figure 2.3 -- optimistic initial values vs realistic
r_opt, o_opt = eps_greedy(0.0, alpha=0.1, Q0=5)     # optimistic, pure greedy
r_real, o_real = eps_greedy(0.1, alpha=0.1, Q0=0)   # realistic, eps-greedy
plot("fig2_3.png", "Fig 2.3 -- optimistic init (Q0=5) vs realistic",
     [("Q0=5, eps=0", r_opt, o_opt), ("Q0=0, eps=0.1", r_real, o_real)])
 
# Figure 2.4 -- UCB vs epsilon-greedy
r_ucb, o_ucb = ucb(2)
r_eg, o_eg = eps_greedy(0.1)
plot("fig2_4.png", "Fig 2.4 -- UCB vs epsilon-greedy",
     [("UCB c=2", r_ucb, o_ucb), ("eps=0.1", r_eg, o_eg)])
 
# Bonus -- gradient bandit
r_g1, o_g1 = gradient_bandit(0.1)
r_g2, o_g2 = gradient_bandit(0.4)
plot("fig2_5_gradient.png", "Gradient bandit (with baseline)",
     [("alpha=0.1", r_g1, o_g1), ("alpha=0.4", r_g2, o_g2)])
 
print("done -- saved fig2_2.png, fig2_3.png, fig2_4.png, fig2_5_gradient.png")