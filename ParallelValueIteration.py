from typing import Set, Callable, Dict
import numpy as np
import pickle 
import multiprocessing as mp
import itertools

# Define the value iteration algorithm as a function
def value_iteration_parallel(S: Set, A: Set, P: Dict, R: Dict, gamma: float, max_iterations: int,
                    bellman_eq: Callable, V_init: Dict = None, theta: float = None) -> Dict:
    '''
    Implements the value iteration algorithm to solve a MDP with given
    set of states and actions, transition probabilities, reward function, 
    and discount factor (gamma)

    Arguments
    -----------
    S: Set of states for the MDP.
    A: Set of actions for the MDP.
    P: A function which calculates P(s'|s,a), the probability of transitioning
        to state s' given we are in state s and execute action a.
    R: A function which calculates R(s,a), the reward obtainined if action a is 
        executed while in state s.
    gamma: Discount factor for calculating the next value function in the MDP.
    max_iterations: maximum number of iterations of value function calculations.
    bellman_eq: a function defining the value function update at each iteration.
    V_init: A dictionary to store the initialised values for all states.
    theta: Threshold value to check convergence of the value function 
        to ensure minimum value function update across all states per iteration.

    Output
    -----------
    output_dict: Dictionary containing the optimal policy dictionary and final value function dictionary for all states
    '''

    # Dictionaries containing value function entries for each state
    Vk = dict([(s, 0) for s in S]) if V_init is None else V_init
    V_next = Vk.copy()

    # Dictionary to store optimal policy for each state
    policy = dict([(s, 0) for s in list(V_init.keys())])

    k = 0 # iteration counter

    while k < max_iterations: # Iteration termination condition
        # print(k, Vk)
        delta = 0       # Factor to check convergence of value function
        k = k+1         # Increment iterations
        vf_inputs = (S, A, P, R, gamma, Vk)
        cl = mp.Pool(3)
        #value_function = value_function_wrapper(bellman_eq)
        s_v_d_triplets = cl.starmap(bellman_eq, list(zip(S, itertools.repeat(S),itertools.repeat(A),itertools.repeat(P),itertools.repeat(R),itertools.repeat(gamma),itertools.repeat(Vk))))
        cl.close()

        # V_next = {s:min(v.values()) for s,v in zip(S,state_action_values)}
        # delta = max([abs(V_next[s]-Vk[s]) for s in S])
        # print(delta)

        Vk = {s:v for s, v, _ in s_v_d_triplets}
        delta = max([d for _, _, d in s_v_d_triplets])
        print(delta)

        # for s in S:     # Update value function for each state in new iteration
            # V_next[s] = min(bellman_eq(s, S, A, P, R, gamma, Vk).values())
            # delta = max(delta, abs(V_next[s] - Vk[s]))
        Vk = V_next.copy()      # Update penultimate value function for all states for next iteration
        if theta != None and delta < theta: # Convergence (termination) condition for value function (if applicable)
            print("Converged!")
            break

    for s in S: # Store optimal policy for each state
        last_value = bellman_eq(s, S, A, P, R, gamma, Vk)
        policy[s] = min(last_value, key = last_value.get)

    return {"optimal_policy": policy, "value_function": V_next}

# def value_function_wrapper(bellman_eq):
#     def value_function(s, extra_info):
#         S, A, P, R, gamma, Vk = extra_info
#         value = min(bellman_eq(s, S, A, P, R, gamma, Vk).values())
#         delta = abs(value - Vk[s])

#         return s, value, delta
    
#     return value_function


# Store value iteration results in a pickle (.pkl) file
def store_results(results, nech, systemLeadtime, capacity, maxA, cb, h, gamma):
    ''' Takes the value iteration results as an input and stores the results in a .pkl file'''
    filename = fr'PolicyResults/{nech}ech_{systemLeadtime}_cap{capacity}_MOQ{maxA}_sl{cb[0]*100/(cb[0]+h[0]):.1f}_gamma{gamma}.pkl'
    pickle.dump(results, open(filename, "wb"))