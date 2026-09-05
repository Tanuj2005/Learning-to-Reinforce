import random
from collections import defaultdict
from enum import IntEnum
from typing import Callable, Dict, List, Set, Tuple
import numpy as np

PlayerSum = int 
DealerCard = int
HasUsableAce = bool 

State = Tuple[ PlayerSum, DealerCard, HasUsableAce ]

class Action( IntEnum ):
    STICK = 0 
    HIT = 1


Policy = Callable[[State], Action]
Trajectory = List[Tuple[State, Action]]

def draw_card() -> int:
    return min( random.randint(1, 13), 10)

def has_usable_ace( hand: List[int] ) -> bool:
    return 1 in hand and sum(hand) + 10 <= 21 

def calculate_hand_value( hand: List[int] ) -> int:
    return sum(hand) + 10 if has_usable_ace( hand ) else sum(hand)

def is_bust( hand: List[int] ) -> bool:
    return calculate_hand_value( hand ) > 21

def play_episode( policy: Policy, 
                 exploring_start: bool = False 
                 ) -> Tuple[ Trajectory, float ]:


    player: List[int] = [draw_card(), draw_card()]
    dealer: List[int] = [draw_card(), draw_card()]

    start_action: Action | None = None

    if exploring_start:

        player = [ random.randint(1, 10), random.randint(1,10)]
        while calculate_hand_value( player ) < 12:
            player.append( draw_card() )
        start_action = random.choice( [Action.STICK, Action.HIT])

    while calculate_hand_value( player ) < 12:
        player.append( draw_card() )

    trajectory: Trajectory = []
    is_first_step: bool = True

    while True:
        state: State = (
            calculate_hand_value(player),
            dealer[0],
            has_usable_ace(player)
        )

        if is_first_step and exploring_start and start_action is not None:
            action = start_action
        else:
            action = policy( state )


        is_first_step = False
        trajectory.append((state, action))


        if action == Action.STICK:
            break

        player.append( draw_card())

        if is_bust( player ):
            return trajectory, -1.0

    while calculate_hand_value( dealer ) < 17:
        dealer.append( draw_card() )


    if is_bust( dealer ):
        return trajectory, 1.0

    player_val = calculate_hand_value( player )
    dealer_val = calculate_hand_value( dealer )

    if player_val > dealer_val :
        reward = 1.0
    elif player_val < dealer_val:
        reward = -1.0
    else:
        reward = 0.0


    return trajectory, reward

def fixed_policy( state: State ) -> Action:

    player_sum, _, _ = state
    return Action.STICK if player_sum >= 20 else Action.HIT 

def mc_prediction(
        policy: Policy,
        num_episodes: int
) -> Dict[ State, float ]:
    value_table: Dict[State, float] = defaultdict( float )
    returns_table: Dict[ State, List[float]] = defaultdict(list)

    for _ in range( num_episodes ):
        trajectory, reward = play_episode( policy )
        visited_states: Set[State] = set()

        for state, _ in trajectory:
            if state not in visited_states:
                visited_states.add( state )
                returns_table[state].append( reward )
                value_table[state] = float( np.mean( returns_table[state]))


    return value_table


def mc_control_exploring_starts(
    num_episodes: int
) -> Tuple[Dict[Tuple[State, Action], float], Dict[State, Action]]:
    """Learns optimal action-value Q(s,a) and policy using Exploring Starts (ES)."""
    q_table: Dict[Tuple[State, Action], float] = defaultdict(float)
    returns_table: Dict[Tuple[State, Action], List[float]] = defaultdict(list)
    policy_table: Dict[State, Action] = {}

    def current_policy(state: State) -> Action:
        return policy_table.get(state, fixed_policy(state))

    for _ in range(num_episodes):
        trajectory, reward = play_episode(current_policy, exploring_start=True)
        visited_state_action_pairs: Set[Tuple[State, Action]] = set()

        for state, action in trajectory:
            pair = (state, action)
            if pair not in visited_state_action_pairs:
                visited_state_action_pairs.add(pair)
                returns_table[pair].append(reward)
                q_table[pair] = float(np.mean(returns_table[pair]))

        # Greedy policy update step
        for state, _ in trajectory:
            q_stick = q_table[(state, Action.STICK)]
            q_hit = q_table[(state, Action.HIT)]
            policy_table[state] = Action.STICK if q_stick >= q_hit else Action.HIT

    return q_table, policy_table




def mc_control_epsilon_soft(
    num_episodes: int, 
    epsilon: float = 0.1
) -> Tuple[Dict[Tuple[State, Action], float], Dict[State, Action]]:
    """Learns optimal policy using epsilon-greedy exploration."""
    q_table: Dict[Tuple[State, Action], float] = defaultdict(float)
    returns_table: Dict[Tuple[State, Action], List[float]] = defaultdict(list)
    policy_table: Dict[State, Action] = {}

    def epsilon_soft_policy(state: State) -> Action:
        if random.random() < epsilon:
            return random.choice([Action.STICK, Action.HIT])
        return policy_table.get(state, fixed_policy(state))

    for _ in range(num_episodes):
        trajectory, reward = play_episode(epsilon_soft_policy)
        visited_state_action_pairs: Set[Tuple[State, Action]] = set()

        for state, action in trajectory:
            pair = (state, action)
            if pair not in visited_state_action_pairs:
                visited_state_action_pairs.add(pair)
                returns_table[pair].append(reward)
                q_table[pair] = float(np.mean(returns_table[pair]))

        for state, _ in trajectory:
            q_stick = q_table[(state, Action.STICK)]
            q_hit = q_table[(state, Action.HIT)]
            policy_table[state] = Action.STICK if q_stick >= q_hit else Action.HIT

    return q_table, policy_table


def mc_control_off_policy(
    num_episodes: int
) -> Tuple[Dict[Tuple[State, Action], float], Dict[State, Action]]:
    """Learns target greedy policy using a completely random behavior policy."""
    q_table: Dict[Tuple[State, Action], float] = defaultdict(float)
    cumulative_weights: Dict[Tuple[State, Action], float] = defaultdict(float)
    target_policy_table: Dict[State, Action] = {}

    def behavior_policy(_: State) -> Action:
        return random.choice([Action.STICK, Action.HIT])  # Uniform random choice

    def target_policy(state: State) -> Action:
        return target_policy_table.get(state, fixed_policy(state))

    for _ in range(num_episodes):
        trajectory, reward = play_episode(behavior_policy)
        weight: float = 1.0

        # Process trajectory in reverse
        for state, action in reversed(trajectory):
            pair = (state, action)
            cumulative_weights[pair] += weight

            # Incremental updates to Q
            q_table[pair] += (weight / cumulative_weights[pair]) * (reward - q_table[pair])

            # Update deterministic greedy target policy
            q_stick = q_table[(state, Action.STICK)]
            q_hit = q_table[(state, Action.HIT)]
            target_policy_table[state] = Action.STICK if q_stick >= q_hit else Action.HIT

            # If behavior action differs from target action, early stop backward loop
            if action != target_policy(state):
                break

            # Behavior probability for uniform random over 2 actions is 0.5
            weight *= 1.0 / 0.5

    return q_table, target_policy_table



if __name__ == "__main__":
    print("1. Running First-Visit MC Prediction...")
    v_10k = mc_prediction(fixed_policy, num_episodes=10_000)
    v_500k = mc_prediction(fixed_policy, num_episodes=500_000)

    print("2. Running MC Control with Exploring Starts...")
    q_es, pi_es = mc_control_exploring_starts(num_episodes=500_000)

    print("3. Running On-Policy Epsilon-Soft MC Control...")
    q_soft, pi_soft = mc_control_epsilon_soft(num_episodes=500_000)

    print("4. Running Off-Policy MC Control via Weighted Importance Sampling...")
    q_off, pi_off = mc_control_off_policy(num_episodes=500_000)

    print("\nExecution completed successfully!")