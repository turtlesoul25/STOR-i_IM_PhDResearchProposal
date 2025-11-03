# Import packages
from typing import List, Set, Dict
import numpy as np
import itertools

def relu(x):
    return max(x, 0)


def create_state_space(capacity: tuple, increment: int, n_ech: int, lead_times: List): 
    ''' Creates a set representing the state space for an n-echelon problem with lead times
    with format (ILs at each site, outstanding orders at each site)'''

    # Possible inventory levels at each site (we assume that all sites have the same capacity)
    IL = set(int(x) for x in np.arange(capacity[0], capacity[1]+1, increment))
    arriving_orders = set(int(x) for x in np.arange(0, capacity[1]+1, increment))

    # Possible set of states
    S = sorted(set((*inv, *arriving) for inv in itertools.product(IL, repeat=n_ech) for arriving in itertools.product(arriving_orders, repeat=sum(lead_times))))

    # Dictionary containing indices for each states (to make value iteration more efficient)
    state_idx = {s: i for i, s in enumerate(S)}    

    return S, state_idx


def create_action_space(capacity: tuple, increment: int, n_ech: int, max_demand: int):
    ''' Creates a set for the action space for an n-echelon inventory problem 
    with format (q1, ..., qn) where qj is the quantity ordered by site j from site j+1''' 

    # Maximum order quantity at each site is lowest_capacity + maximum demand so that backlogs can be cleared
    order_set = set(int(x) for x in np.arange(0, relu(-capacity[0]) + max_demand + 1, increment))

    # Possible actions
    A = sorted(set(order_tuple for order_tuple in itertools.product(order_set, repeat=n_ech)))

    # Dictionary containing indices for each action (for value iteration)
    action_idx = {a: i for i, a in enumerate(A)}

    return A, action_idx


def create_P(S: Set, A: Set, state_idx: Dict, action_idx: Dict, demand_distribution: Dict, capacity: Dict, lead_times: List):
    '''Creates an array containing transition probabilities from s to s' under action a
    for a centralised multi-echelon serial system with lead times'''

    def prob_trans(s, a, s_next):
        '''Calculates transition probability from s to s' under action a'''
        prob = 0

        # Check next inventory level for warehouse
        s_next_w = s[1] + a[1] - a[0]
        s_next_w = max(capacity[0], min(s_next_w, capacity[1]))  # Truncate W IL which is outside of backlog/capacity range

        # Check outstanding orders
        q_sent_to_DC = min(relu(s[1])+a[1], relu(-s[1]) + a[0])
        arr_orders_DC = s[3:(2+lead_times[0])] + (q_sent_to_DC,) 

        if (s_next_w, ) + arr_orders_DC != s_next[1:]:
            return prob

        # Check next inventory level for DC
        for d in demand_distribution.keys():
            s_next_DC = s[0] + s[2] - d
            if capacity[0] < s_next_DC < capacity[1] and s_next_DC == s_next[0]:   # Non truncated state
                prob = demand_distribution.get(d, 0)
                return prob

            elif s_next_DC >= capacity[1] and s_next[0] == capacity[1]: # truncated state above capacity
                prob = sum(demand_distribution[dem] for dem in demand_distribution if s[0] + s[2] - dem >= capacity[1])
                return prob
            
            elif s_next_DC <= capacity[0] and s_next[0] == capacity[0]: # truncated state below backlog limit
                prob = sum(demand_distribution[dem] for dem in demand_distribution if s[0] + s[2] - dem <= capacity[0])
                return prob
            
        
        return prob
            
    # Array to store transition probabilities for all combinations of s, a, s'
    P_array = np.zeros((len(S), len(A), len(S)))

    for s in S: # for each state s
        s_idx = state_idx[s]
        for a in A: # for each action a
            a_idx = action_idx[a]
            for s_next in S: # for each new state s'
                sp_idx = state_idx[s_next]
                # Calculate and store transition probability
                P_array[s_idx, a_idx, sp_idx] = prob_trans(s, a, s_next)

    return P_array

def create_R(S: Set, A: Set, state_idx: Dict, action_idx: Dict, demand_distribution: Dict,
             hold_costs: List, backlog_costs: List, lead_times: List, ProbTrans_Array: np.ndarray = None):
    '''
    Creates an array containing the reward obtained under action a chosen at
    state s for a centralised multi-echelon serial system with lead times.
    '''

    # def cost_function(s, a, sp):
    #     '''Calculates cost'''


    def expected_cost_function(s, a):
        '''Calculates expected cost incurred if action a is taken at state s'''
        
        # Store possible final DC ILs with probability and final warehouse ILs
        dc_il_next = [(s[0] + s[2] - dt, prob) for dt, prob in demand_distribution.items()] 
        w_il_next = s[1] + a[1] - a[0]  ##### Need to change if warehouse has nonzero lead times
        
        warehouse_cost = hold_costs[1]*relu(w_il_next) + backlog_costs[1]*relu(-w_il_next)
        dc_cost = hold_costs[0]*sum(relu(il)*prob for (il, prob) in dc_il_next) + backlog_costs[0]*sum(relu(-il)*prob for (il, prob) in dc_il_next)

        return warehouse_cost + dc_cost
    
    R_array = np.zeros((len(S), len(A)))

    for s in S: # for each state s
        s_idx = state_idx[s]
        for a in A: # for each action a
            a_idx = action_idx[a]
            R_array[s_idx, a_idx] = expected_cost_function(s, a) # calculate reward for taking action a at state s

    return R_array
        
        ##### WORK ON REWARD FUNCTION FROM HERE
        

def cL_value_update_func(state_idx: Dict, action_idx: Dict, capacity: tuple, demand_distribution: Dict):
    max_demand = max(demand_distribution.keys())
    def bellman_eq_2cL(s, S, A, P, R, gamma, Vk):
        ''' Calculates the values from taking each action at state s '''
        s_idx = state_idx[s]

        # Ordering decisions should ensure that site capacity is not exceeded
        values = dict((a, 0) for a in A if s[0]+a[0] <= min(capacity[1], max_demand) and s[1]+a[1] <= min(capacity[1], max_demand))

        if not values: # if no possible ordering decisions, then no units need to be ordered
            values = {(0, 0): 0}
        
        for a in values.keys():
            a_idx = action_idx[a]
            values[a] = R[s_idx, a_idx] + gamma*sum([P[s_idx, a_idx, state_idx[sp]]*Vk[sp] for sp in S])
        return values

    return bellman_eq_2cL


# # Code to check transition probabilities
# for s in S:
#     s_idx = state_idx[s]
#     for a in A:
#         a_idx = action_idx[a]
#         if sum(prob_trans[s_idx, a_idx, state_idx[s_next]] for s_next in S) != 1:
#             print(s, a)
#             print([(s_next, prob_trans[s_idx, a_idx, state_idx[s_next]]) for s_next in S if prob_trans[s_idx, a_idx, state_idx[s_next]] > 0])
            
