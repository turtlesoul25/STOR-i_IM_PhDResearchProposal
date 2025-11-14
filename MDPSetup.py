# Import packages
from typing import List, Set, Dict
import numpy as np
import itertools

def relu(x):
    return max(x, 0)


def create_state_space(capacity: tuple, increment: int, n_ech: int, max_demand: int, lead_times: List): 
    ''' Creates a set representing the state space for an n-echelon problem with lead times
    with format (ILs at each site, outstanding orders at each site)'''

    # Possible inventory levels at each site (we assume that all sites have the same capacity)
    IL = set(int(x) for x in np.arange(capacity[0], capacity[1]+1, increment))
    arriving_orders = set(int(x) for x in np.arange(0, relu(-capacity[0]) + max_demand + 1, increment))
    ################ What will be the max size of arriving orders????
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


def create_P(S: Set, A: Set, state_idx: Dict, action_idx: Dict, demand_distribution: Dict, capacity: Dict, n_ech: int, lead_times: List):
    '''Creates an array containing transition probabilities from s to s' under action a
    for a centralised multi-echelon serial system with lead times'''

    def prob_trans(s, a, s_next):
        '''Calculates transition probability from s to s' under action a'''
        prob = 0

        # Check next inventory level for warehouse
        next_w_il = s[1] + a[1] - a[0] if lead_times[1] == 0 else s[1] + s[n_ech + lead_times[0]] - a[0]
        next_w_il = max(capacity[0], min(next_w_il, capacity[1]))  # Truncate W IL which is outside of backlog/capacity range

        if next_w_il != s_next[1]:
            return prob
        
        # Check outstanding orders for warehouse
        arr_orders_w = () if lead_times[1] == 0 else s[n_ech + lead_times[0] + 1 : n_ech + lead_times[0] + lead_times[1]] + (a[1], )
        if lead_times[1] > 0 and arr_orders_w != s_next[n_ech + lead_times[0]: n_ech + lead_times[0] + lead_times[1]]:
            return prob
        

        # Check outstanding orders for DC
        q_sent_to_DC = min(a[0] + relu(-s[1]), relu(s[1]) + a[1] if lead_times[1] == 0 else relu(s[1]) + s[n_ech + lead_times[0]])
        arr_orders_dc = () if lead_times[0] == 0 else s[n_ech + 1 : n_ech + lead_times[0]] + (q_sent_to_DC, )
        # print(arr_orders_dc, s_next[n_ech: n_ech + lead_times[0]])
        if lead_times[0] > 0 and arr_orders_dc != s_next[n_ech: n_ech + lead_times[0]]:
            return prob
        
        # Check next inventory level for DC
        dc_il_pre_demand = s[0] + q_sent_to_DC if lead_times[0] == 0 else s[0] + s[n_ech]

        if s_next[0] > dc_il_pre_demand:  # Next DC IL cannot be greater than DC IL pre-demand
            return prob
        
        for d in demand_distribution.keys():
            next_dc_il = dc_il_pre_demand - d

            if capacity[0] < next_dc_il < capacity[1] and next_dc_il == s_next[0]:   # non-truncated state
                prob = demand_distribution.get(d, 0)
                return prob
            
            elif next_dc_il >= capacity[1] and s_next[0] == capacity[1]:      # truncated state above capacity
                prob = sum(demand_distribution[dem] for dem in demand_distribution if dc_il_pre_demand - dem >= capacity[1])
                return prob
            
            elif next_dc_il <= capacity[0] and s_next[0] == capacity[0]: # truncated state below backlog limit
                prob = sum(demand_distribution[dem] for dem in demand_distribution if dc_il_pre_demand - dem <= capacity[0])
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
             hold_costs: List, backlog_costs: List, n_ech: int, lead_times: List):
    '''
    Creates an array containing the reward obtained under action a chosen at
    state s for a centralised multi-echelon serial system with lead times.
    '''

    def expected_cost_function(s, a):
        '''Calculates expected cost incurred if action a is taken at state s'''
        
        # Store possible final DC ILs with probability and final warehouse ILs
        w_il_next = s[1] + a[1] - a[0] if lead_times[1] == 0 else s[1] + s[n_ech + lead_times[0]] - a[0]
        if lead_times[0] == 0:
            q_sent_to_DC = min(a[0] + relu(-s[1]), relu(a[0] + w_il_next))
            dc_il_next = [(s[0] + q_sent_to_DC - dt, prob) for dt, prob in demand_distribution.items()]
        else:
            dc_il_next = [(s[0] + s[n_ech] - dt, prob) for dt, prob in demand_distribution.items()] 
        
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
        
        

def cL_value_update_func(state_idx: Dict, action_idx: Dict, capacity: tuple, demand_distribution: Dict):
    max_demand = max(demand_distribution.keys())
    def bellman_eq_2cL(s, S, A, P, R, gamma, Vk):
        ''' Calculates the values from taking each action at state s '''
        s_idx = state_idx[s]

        # Ordering decisions should ensure that site capacity is not exceeded
        values = dict((a, 0) for a in A if s[0]+a[0] <= min(capacity[1], max_demand) and s[1]+a[1] <= capacity[1])

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
            
