# Import packages
from typing import List, Set, Dict, Callable
import numpy as np
import itertools

def relu(x):
    return max(x, 0)


def create_state_space(capacity: tuple, increment: int, n_ech: int, max_demand: int, lead_times: List): 
    ''' Creates a set representing the state space for an n-echelon problem with lead times
    with format (ILs at each site, outstanding orders at each site)'''

    # Possible inventory levels at each site (we assume that all sites have the same capacity)
    IL = set(int(x) for x in np.arange(capacity[0], capacity[1]+1, increment))
    arriving_orders = set(int(x) for x in np.arange(0, relu(-capacity[0]) + capacity[1] + 1, increment))
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
    order_set = set(int(x) for x in np.arange(0, relu(-capacity[0]) + capacity[1] + 1, increment))

    # Possible actions
    A = sorted(set(order_tuple for order_tuple in itertools.product(order_set, repeat=n_ech)))

    # Dictionary containing indices for each action (for value iteration)
    action_idx = {a: i for i, a in enumerate(A)}

    return A, action_idx


def create_P(S: Set, A: Set, state_idx: Dict, action_idx: Dict, demand_distribution: Dict, capacity: tuple, n_ech: int, lead_times: List):
    '''Creates an array containing transition probabilities from s to s' under action a
    for a centralised multi-echelon serial system with lead times'''

    def prob_trans(s, a):
        ''' Calculates list of possible next states and defines probability of transition to those states'''
        state_trans_probs = dict()

        # Calculate next inventory level for warehouse
        next_w_il = s[1] + a[1] - a[0] if lead_times[1] == 0 else s[1] + s[n_ech + lead_times[0]] - a[0]
        next_w_il = max(capacity[0], min(next_w_il, capacity[1]))  # Truncate W IL which is outside of backlog/capacity range

        # Calculate next arriving warehouse orders
        arr_orders_w = () if lead_times[1] == 0 else s[n_ech+lead_times[0]+1 : n_ech+lead_times[0]+lead_times[1]] + (a[1], )

        # Calculate next arriving orders for DC
        q_sent_to_DC = min(a[0] + relu(-s[1]), relu(s[1]) + a[1] if lead_times[1] == 0 else relu(s[1]) + s[n_ech + lead_times[0]])
        arr_orders_dc = () if lead_times[0] == 0 else s[n_ech + 1 : n_ech + lead_times[0]] + (q_sent_to_DC, )

        # Calculate next inventory level for DC
        dc_il_pre_demand = s[0] + q_sent_to_DC if lead_times[0] == 0 else s[0] + s[n_ech]
        
        for d, prob in demand_distribution.items():
            next_dc_il = dc_il_pre_demand - d
            next_dc_il = max(capacity[0], min(next_dc_il, capacity[1]))  # Truncate DC IL which is outside of backlog/capacity range
            next_state = (next_dc_il, next_w_il, ) + arr_orders_dc + arr_orders_w 
            state_trans_probs[next_state] = state_trans_probs.get(next_state, 0) + prob

        return state_trans_probs
    
    transitions = {(s,a): prob_trans(s, a) for a in A for s in S}
    
    return transitions
    
        


    


def create_R(S: Set, A: Set, state_idx: Dict, action_idx: Dict, P: Callable, demand_distribution: Dict,
             hold_costs: List, backlog_costs: List, capacity: tuple, n_ech: int, lead_times: List):
    '''
    Creates an array containing the reward obtained under action a chosen at
    state s for a centralised multi-echelon serial system with lead times.
    '''

    def expected_cost_function(s, a):
        '''Calculates expected cost incurred if action a is taken at state s'''

        # Calculate final W IL
        next_w_il = s[1] + a[1] - a[0] if lead_times[1] == 0 else s[1] + s[n_ech + lead_times[0]] - a[0]
        
        # Store possible final DC ILs with probability and final warehouse ILs
        if lead_times[0] == 0:
            q_sent_to_DC = min(a[0] + relu(-s[1]), relu(s[1]) + a[1]) if lead_times[1] == 0 else min(a[0] + relu(-s[1]), relu(s[1]) + s[n_ech + lead_times[0]])
            dc_il_next = [(s[0] + q_sent_to_DC - dt, prob) for dt, prob in demand_distribution.items()]
        else:
            dc_il_next = [(s[0] + s[n_ech] - dt, prob) for dt, prob in demand_distribution.items()] 
        
        # Calculate costs at each site
        warehouse_cost = (hold_costs[1]*relu(next_w_il)) + (backlog_costs[1]*relu(-next_w_il))
        dc_cost = (hold_costs[0]*sum(relu(il)*prob for (il, prob) in dc_il_next)) + (backlog_costs[0]*sum(relu(-il)*prob for (il, prob) in dc_il_next))

        return warehouse_cost + dc_cost # Returns total expected system cost
    
    costs = {(s,a): expected_cost_function(s, a) for a in A for s in S}
    return costs
        
        

def cL_value_update_func(S: Set, A: Set, state_idx: Dict, action_idx: Dict, 
                         P: Callable, capacity: tuple, demand_distribution: Dict, n_ech: int, lead_times: List):
    max_demand = max(demand_distribution.keys())
    def bellman_eq_2cL(s, S, A, P, R, gamma, Vk, verbose = False):
        ''' Calculates the values from taking each action at state s '''

        # Ordering decisions should ensure that site maximum inventory level is not exceeded to ensure truncation works correctly
        values = dict((a, 0) for a in A if s[0]+a[0]+sum(s[n_ech:n_ech+lead_times[0]]) <= capacity[1]+(lead_times[0]*max_demand) and s[1]+a[1]+sum(s[n_ech+lead_times[0]:n_ech+lead_times[0]+lead_times[1]]) <= capacity[1]+(lead_times[1]*max_demand))
        # values = dict((a, 0) for a in A)
        if not values: # if no possible ordering decisions, then no units need to be ordered
            values = {(0, 0): 0}
        
        for a in values.keys():
            values[a] = R[s, a] + gamma*sum(Prob * Vk[sp] for sp, Prob in P[s,a].items())
        
        if verbose:
            return values
        
        min_value = min(values.values())
        # print(s, min_value, min(values, key=values.get))
        return min_value, min(values, key=values.get),  abs(Vk[s] - min_value)

    return bellman_eq_2cL

# def bellman_eq_2cL(s, S, A, P, R, gamma, Vk):
#         ''' Calculates the values from taking each action at state s '''
#         # s_idx = state_idx[s]

#         # Ordering decisions should ensure that site capacity is not exceeded
#         # values = dict((a, 0) for a in A if s[0]+a[0] <= min(capacity[1], max_demand) and s[1]+a[1] <= capacity[1])
#         values = dict((a, 0) for a in A)
#         if not values: # if no possible ordering decisions, then no units need to be ordered
#             values = {(0, 0): 0}
        
#         for a in values.keys():
#             # a_idx = action_idx[a]
#             values[a] = R[s, a] + gamma*sum(Prob * Vk[sp] for sp, Prob in P[s,a].items())
        
#         value = min(values.values())
#         delta = abs(value - Vk[s])
#         return s, min(values.values()), delta


# # Code to check transition probabilities
# for s in S:
#     for a in A:
#         if sum(P[s, a].get(s_next, 0) for s_next in S) != 1:
#             print(s, a)
