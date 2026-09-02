import numpy as np

GOAL = 100
states = np.arange( 1, GOAL )
p_h = 0.4
gamma = 1.0

def actions( s ):
    return range( 1, min( s, GOAL - s) + 1 )

def transitions( s, a ):
    win = ( p_h, s + a, 1.0 if s+a == GOAL else 0.0 )
    lose = ( 1-p_h, s-a, 0.0 )
    return [ win, lose ]

def value_iteration( tol = 1e-9 ):
    V = np.zeros( GOAL + 1 )
    sweeps = 0 
    while True:
        delta = 0 
        for s in states:
            best = max(sum(p*(r + gamma*V[s2]) for p,s2,r in transitions(s,a)) for a in actions(s))
            delta = max( delta, abs( best - V[s]))
            V[s] = best 
        sweeps += 1
        if delta < tol:
            break 

    policy = np.zeros( GOAL + 1, dtype = int )

    for s in states:
        qs = [(a, sum(p*(r + gamma*V[s2]) for p,s2,r in transitions(s,a))) for a in actions(s)]
        best_q = max( q for _,q in qs )
        policy[s] = min( a for a,q in qs if abs(q-best_q) < 1e-5)
    return V, policy, sweeps

def policy_evaluation( policy, tol = 1e-9):
    V = np.zeros( GOAL + 1 )
    while True:
        delta = 0
        for s in states:
            a = policy[s]
            v_new = sum( p*( r + gamma*V[s2]) for p,s2,r in transitions(s,a))
            delta = max( delta, abs( v_new - V[s]))
            V[s] = v_new
        if delta < tol:
            return V 

def policy_improvement(V):
    policy = np.zeros( GOAL + 1, dtype= int )
    stable = True 
    for s in states:
        qs = [(a, sum(p*(r + gamma*V[s2]) for p,s2,r in transitions(s,a))) for a in actions(s)]
        best_q = max( q for _, q in qs)
        policy[s] = min( a for a, q in qs if abs(q-best_q) < 1e-5)

    return policy

def policy_iteration():
    policy = np.ones( GOAL + 1, dtype=int)
    iters = 0
    while True:
        V = policy_evaluation( policy )
        new_policy = policy_improvement(V)
        iters += 1
        if np.array_equal( new_policy, policy):
            return V, new_policy, iters 
        policy = new_policy

V_vi, pi_vi, sweeps_vi = value_iteration()
V_pi, pi_pi, iters_pi = policy_iteration()

print(f"value iteration converged in {sweeps_vi} sweeps")
print(f"policy iteration converged in {iters_pi} policy-improvement rounds")
print("V matches:", np.allclose(V_vi, V_pi, atol=1e-4))
print("policies match:", np.array_equal(pi_vi, pi_pi))

# checkpoint: plot pi_vi[1:100] — should show the jagged, non-monotonic staking pattern of Fig 4.6
import matplotlib.pyplot as plt
plt.step(states, pi_vi[1:GOAL], where='mid')
plt.xlabel("Capital"); plt.ylabel("Final stake"); plt.title(f"Optimal policy (p_h={p_h})")
plt.show()