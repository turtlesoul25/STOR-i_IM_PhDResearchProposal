# Import packages
from typing import Set, Callable, Dict, List
import numpy as np

def relu(x):
    ''' Returns x if positive and 0 otherwise'''
    return max(x, 0)


def generate_demand(demand_distribution: Dict) -> float:
    ''' Generates random retailer demand for the current 
    time period from a given demand distribution
    
    Input:
        demand_distribution: a dictionary where keys represent demand quantity
                             and values represent probability of the demand quantity
    Returns a demand quantity by choosing a key from the dictionary'''

    return np.random.choice([*demand_distribution], p = [*demand_distribution.values()])


def simulate_1ech_nolead(num_periods: int, il0: int, h: float, p: float, 
                     demand_distribution: Dict, policy: Dict):
    '''
    Simulates a single echelon supply chain containing a warehouse supplying 
    inventory to a customer with stochastic demand without any lead time
    --------
    Inputs: 
        num_periods: length of simulation horizon
        il0: initial inventory level at warehouse
        h: unit holding cost at warehouse
        p: unit backlog penalty cost at warehouse
        demand_distribution: dictionary describing probability distribution of customer demand
        policy: warehouse ordering policy
        
    Returns:
        states_visited: List with history of ILs at warehouse from each time period of the simulation
        ILs_pre_demand: List with history of ILs after orders arrive and before customer demand is realised
        costs: List with history of costs incurred at warehouse in each time period
        demands: List with history of customer demand from each time period
        total_cost: Total cost incurred by warehouse across simulation horizon
    '''



    il, total_cost = il0, 0   # Initial warehouse inventory and initial total cost during simulation
    states_visited = [il0]    # List to store all states visited in simulation
    ILs_pre_demand = []  # List to store ILs after orders arrive and before customer demand is realised
    costs = []      # List to store costs incurred in each time period
    demands = []              # List to store demands
    actions = []              # List to store order decision made by warehouse in each time period

    for t in range(num_periods): 
        action = policy[il]       # choose optimal order quantity from policy
        il += action              # IL at warehouse increases by order quantity due to no lead time
        ILs_pre_demand.append(il) # Add warehouse IL after replenishment order arrives and before customer demand is realised 
        d = int(generate_demand(demand_distribution))  # Customer demand is realised
        il -= d                # Warehouse sends inventory to customer (or updates backlogs)
        period_cost = max(il, 0)*h + max(-il, 0)*p   # Cost incurred in the current time period
        total_cost += period_cost                    # Update total costs incurred in the system across simulation horizon so far
        
        # Add IL, cost, demand and action from the current time period to histories
        states_visited.append(il)
        costs.append(period_cost)
        demands.append(d)
        actions.append(action)

    return states_visited, ILs_pre_demand, costs, demands, total_cost



def simulate_1ech_constantlead(num_periods: int, ip0: tuple, h: float, p: float, 
                     demand_distribution: Dict, policy: Dict):
    '''
    Simulates a single echelon supply chain containing a warehouse supplying 
    inventory to a customer with stochastic demand with constant, deterministic lead time
    --------
    Inputs: 
        num_periods: length of simulation horizon
        ip0: tuple describing initial inventory position at warehouse
        h: unit holding cost at warehouse
        p: unit backlog penalty cost at warehouse
        demand_distribution: dictionary describing probability distribution of customer demand
        lead_time: lead time for orders placed by warehouse with production facility
        policy: warehouse ordering policy
        
    Returns:
        states_visited: List with history of ILs at warehouse from each time period of the simulation
        ILs_pre_demand: List with history of ILs after orders arrive and before customer demand is realised
        costs: List with history of costs incurred at warehouse in each time period
        demands: List with history of customer demand from each time period    
        total_cost: Total cost incurred by warehouse across simulation horizon
    '''

    ip, total_cost = ip0, 0   # Initial warehouse inventory position and initial total cost during simulation
    states_visited = [ip0]    # List to store all states visited in simulation
    ILs_pre_demand = []  # List to store ILs after orders arrive and before customer demand is realised
    costs = []      # List to store costs incurred in each time period
    demands = []              # List to store demands
    actions = []              # List to store order decision made by warehouse in each time period

    for t in range(num_periods): 
        action = policy[ip]      # choose optimal order quantity from policy
        new_il = ip[0] + ip[-1]  # update IL after arriving order and before customer demand is realised
        new_outstanding_orders = (action, ) + ip[1:-1]       # Update warehouse outstanding orders with new order and remove arrived order
        ILs_pre_demand.append(new_il)                        # Add IL at warehouse before customer demand is realised
        d = int(generate_demand(demand_distribution))        # Customer demand is realised
        new_il -= d                                          # Warehouse sends inventory to customer (or updates backlogs)
        period_cost = max(new_il, 0)*h + max(-new_il, 0)*p   # Cost incurred in the current time period
        total_cost += period_cost                            # Update total costs incurred in the system across simulation horizon so far
        ip = (new_il, ) + new_outstanding_orders             # Update inventory position after customer demand is realised and before next time period begins

        # Add IL, cost, demand and action from the current time period to histories
        states_visited.append(ip)
        costs.append(period_cost)
        demands.append(d)
        actions.append(action)

    return states_visited, ILs_pre_demand, costs, demands, total_cost



def simulate_2ech_nolead(num_periods: int, ils0: tuple, h: List, p: List, 
                         demand_distribution: Dict, policy: Dict):
    
    dc_il, w_il, total_cost = ils0[0], ils0[1], 0
    states_visited = [ils0]
    ILs_pre_demand, costs, demands, actions = [], [], [], []

    for t in range(num_periods):
        # Step 1
        dc_q, w_q = policy[(dc_il, w_il)]   # Obtain optimal order quantity for DC and warehouse
        dc_il += min(relu(w_il) + w_q, relu(-w_il) + dc_q)   # Updates DC IL with replenishment order based on what warehouse can send
        w_il += (w_q - dc_q)                # Update warehouse IL once replenishment order arrives and order is sent to DC
        ILs_pre_demand.append((dc_il, w_il))                 # Store ILs at warehouse and DC before customer demand is realised

        # Step 2
        d = int(generate_demand(demand_distribution))   # Customer demand is realised
        dc_il -= d                                      # Dc sends order to customer

        # Steps 3 and 4
        period_cost = relu(dc_il)*h[0] + relu(w_il)*h[1] + relu(-dc_il)*p[0] + relu(-w_il)*p[1]  # Costs incurred in the time period
        total_cost += period_cost      # Total system costs incurred across simulation time horizon so far

        # Add ILs, costs, demands, actions to the list of histories
        states_visited.append((dc_il, w_il))
        costs.append(period_cost)
        demands.append(d)
        actions.append((dc_q, w_q))

    return states_visited, ILs_pre_demand, costs, demands, actions, total_cost

def clean_decentralised_2ech_policy(policy, S, Q_max):
    '''
    Reformats a decentralised policy to include decisions of warehouse and DC for each IL tuple
    
    Inputs: 
        policy: Dictionary containing the warehouse order policy for each IL tuple at (DC, warehouse)
        S: order up to level followed by DC
        Q_max: maximum allowed order quantity for DC
        
    Outputs:
        full_policy: Dictionary containing (DC, warehouse) optimal order quantity as the value 
            for each (DC, warehouse) IL combination
    '''

    full_policy = {(dc_il, w_il): (min(relu(S - dc_il), Q_max), w_q) for (dc_il, w_il), w_q in policy.items()}
    return full_policy
    