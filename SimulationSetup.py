# Import packages
from typing import Set, Callable, Dict, List
import numpy as np
import matplotlib.pyplot as plt

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


def simulate_1ech_nolead(num_periods: int, gamma: float, il0: int, h: float, p: float, 
                     demand_distribution: Dict, policy: Dict, tol: float):
    '''
    Simulates a single echelon supply chain containing a warehouse supplying 
    inventory to a customer with stochastic demand without any lead time
    --------
    Inputs: 
        num_periods: length of simulation horizon
        gamma: discount factor for costs 
        il0: initial inventory level at warehouse
        h: unit holding cost at warehouse
        p: unit backlog penalty cost at warehouse
        demand_distribution: dictionary describing probability distribution of customer demand
        policy: warehouse ordering policy
        tol: convergence threshold (stopping critertion) to detect negligible change in costs
        
    Returns:
        states_visited: List with history of ILs at warehouse from each time period of the simulation
        ILs_pre_demand: List with history of ILs after orders arrive and before customer demand is realised
        costs: List with history of costs incurred at warehouse in each time period
        demands: List with history of customer demand from each time period
        total_cost: Total discounted cost incurred by warehouse across simulation horizon
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
        period_cost = relu(il)*h + relu(-il)*p   # Cost incurred in the current time period
        total_cost += (gamma**t)*period_cost                    # Update total costs incurred in the system across simulation horizon so far
        
        # Add IL, cost, demand and action from the current time period to histories
        states_visited.append(il)
        costs.append(period_cost)
        demands.append(d)
        actions.append(action)

        if period_cost*(gamma**t) < tol:     # Convergence stopping criterion for negligible cost increase
            break

    return states_visited, ILs_pre_demand, costs, demands, total_cost



def simulate_1ech_constantlead(num_periods: int, gamma: float, ip0: tuple, h: float, p: float, 
                     demand_distribution: Dict, policy: Dict, tol: float):
    '''
    Simulates a single echelon supply chain containing a warehouse supplying 
    inventory to a customer with stochastic demand with constant, deterministic lead time
    --------
    Inputs: 
        num_periods: length of simulation horizon
        gamma: discount factor for costs
        ip0: tuple describing initial inventory position at warehouse
        h: unit holding cost at warehouse
        p: unit backlog penalty cost at warehouse
        demand_distribution: dictionary describing probability distribution of customer demand
        lead_time: lead time for orders placed by warehouse with production facility
        policy: warehouse ordering policy
        tol: convergence threshold (stopping critertion) to detect negligible change in costs
        
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
        period_cost = relu(new_il)*h + relu(-new_il)*p       # Cost incurred in the current time period
        total_cost += (gamma**t)*period_cost                 # Update total costs incurred in the system across simulation horizon so far
        ip = (new_il, ) + new_outstanding_orders             # Update inventory position after customer demand is realised and before next time period begins

        # Add IL, cost, demand and action from the current time period to histories
        states_visited.append(ip)
        costs.append(period_cost)
        demands.append(d)
        actions.append(action)

        if period_cost*(gamma**t) < tol:     # Convergence stopping criterion for negligible cost increase
            break

    return states_visited, ILs_pre_demand, costs, demands, total_cost



def simulate_2ech_nolead(num_periods: int, gamma: float, ils0: tuple, h: List, p: List, 
                         demand_distribution: Dict, policy: Dict, tol: float):
    
    dc_il, w_il, total_cost = ils0[0], ils0[1], 0
    states_visited = [ils0]
    ILs_pre_demand, costs, demands, actions = [], [], [], []
    dc_il_bounds = [min(ils[0] for ils in policy), max(ils[0] for ils in policy)]
    w_il_bounds = [min(ils[1] for ils in policy), max(ils[1] for ils in policy)]

    for t in range(num_periods):
        # Step 1
        # print(f"start ILs = {(dc_il, w_il)}")
        dc_q, w_q = policy[(dc_il, w_il)]   # Obtain optimal order quantity for DC and warehouse
        # print(f"Action = {(dc_q, w_q)}")
        dc_il += min(relu(w_il) + w_q, relu(-w_il) + dc_q)   # Updates DC IL with replenishment order based on what warehouse can send
        w_il += (w_q - dc_q)                # Update warehouse IL once replenishment order arrives and order is sent to DC
        ILs_pre_demand.append((dc_il, w_il))                 # Store ILs at warehouse and DC before customer demand is realised

        # Step 2
        d = int(generate_demand(demand_distribution))   # Customer demand is realised
        # print(f"Demand = {d}")
        dc_il -= d                                      # DC sends order to customer

        # Steps 3 and 4
        period_cost = relu(dc_il)*h[0] + relu(w_il)*h[1] + relu(-dc_il)*p[0] + relu(-w_il)*p[1]  # Costs incurred in the time period
        # print(f"Period cost = {period_cost}, discounted cost = {(gamma**t)*period_cost}")
        total_cost += (gamma**t)*period_cost      # Total system costs incurred across simulation time horizon so far

        # Check IL bounds
        if not dc_il_bounds[0] <= dc_il <= dc_il_bounds[1]:  # If DC_il outside allowed limit
            dc_il = max(min(dc_il, dc_il_bounds[1]), dc_il_bounds[0])
        if not w_il_bounds[0] <= w_il <= w_il_bounds[1]:  # If w_il outside allowed limit
            w_il = max(min(w_il, w_il_bounds[1]), w_il_bounds[0])

        # Add ILs, costs, demands, actions to the list of histories
        states_visited.append((dc_il, w_il))
        costs.append(period_cost)
        demands.append(d)
        actions.append((dc_q, w_q))

        if 0 < period_cost*(gamma**t) < tol:  # Convergence stopping criterion for negligible cost increase
            # print(f"Converged at {t} when starting with {ils0}")
            break

    return states_visited, ILs_pre_demand, costs, demands, actions, total_cost



def simulate_2ech_constlead(num_periods: int, gamma: float, ips0: tuple, h: List, p: List, 
                            demand_distribution: Dict, lead_times: List, policy: Dict, tol: float):
    '''
    Simulates a 2 echelon supply chain containing a warehouse and a DC
    where the DC supplies inventory to a customer with stochastic demand with constant lead times
    -------
    Inputs: 
        num_periods: length of simulation_horizon
        gamma: discount factor for costs
        il0: initial inventory level at warehouse
        h: unit holding cost at warehouse
        p: unit backlog penalty cost at warehouse
        demand_distribution: dictionary describing the probability distribution for customer demand
        lead_times: list containing the lead time at each site in the supply chain
        policy: ordering policy at both sites
        tol: convergence threshold (stopping criterion) to detect negligible change in costs
        
    Returns: 
        states_visited: List with history of ILs at both sites from each time period of the simulation
        ILs_pre_demand List with history of ILs after orders arrive and before customer demand is realised
        costs: List with history of supply chain costs incurred in each time period
        demands: List with history of customer demand from each time period
        total_cost: Total discounted cost incurred by warehouse across simulation horizon
    '''

    dc_il, w_il, total_cost = ips0[0], ips0[1], 0
    dc_arr_orders = ips0[2 : 2+lead_times[0]] if lead_times[0] > 0 else ()
    w_arr_orders = ips0[2+lead_times[0]:] if lead_times[1] > 0 else ()

    states_visited = [ips0] 
    ILs_pre_demand, IPs_pre_demand, costs, demands, actions = [], [], [], [], []
    dc_il_bounds = [min(ils[0] for ils in policy), max(ils[0] for ils in policy)]
    w_il_bounds = [min(ils[1] for ils in policy), max(ils[1] for ils in policy)]

    for t in range(num_periods):
        # Step 1: warehouse and DC place orders according to policy
        dc_q, w_q = policy[(dc_il, w_il, ) + dc_arr_orders + w_arr_orders]
        # IPs_pre_demand.append((dc_il + dc_q + sum(dc_arr_orders), w_il + w_q + sum(w_arr_orders)))
        # print(f"start ILs = {(dc_il, w_il)}")
        # print(f"action = {(dc_q, w_q)}")

        # Step 2: warehouse receives order, ships to DC, then DC receives order
        # print(states_visited[-1], dc_q, w_q)
        q_sent_to_dc = min(dc_q + relu(-w_il), relu(w_il) + (w_q if lead_times[1] == 0 else w_arr_orders[0]))                  # quantity shipped out by warehouse
        w_il += (w_q - dc_q) if lead_times[1] == 0 else (w_arr_orders[0] - dc_q)    # update warehouse IL                                        # update warehouse IL
        dc_il += q_sent_to_dc if lead_times[0] == 0 else dc_arr_orders[0]       # DC receives shipment
        dc_arr_orders = dc_arr_orders[1:] + (q_sent_to_dc, ) if lead_times[0] > 0 else ()  # update outstanding orders for DC
        w_arr_orders = w_arr_orders[1:] + (w_q, ) if lead_times[1] > 0 else ()           # update outstanding orders for warehouse
        IPs_pre_demand.append((dc_il + sum(dc_arr_orders), w_il + sum(w_arr_orders)))
        ILs_pre_demand.append((dc_il, w_il))

        # Step 3: Customer demand is realised
        d = int(generate_demand(demand_distribution))     # customer demand realised
        dc_il -= d 
        # print(f"Demand = {d}, final ILs = {dc_il, w_il}")                                       # Update DC IL after customer demand

        # Step 4: Charge costs
        period_cost = relu(dc_il)*h[0] + relu(w_il)*h[1] + relu(-dc_il)*p[0] + relu(-w_il)*p[1]  # Costs incurred in the time period
        # print(f"Period cost = {period_cost}, discounted cost = {(gamma**t)*period_cost}")
        total_cost += (gamma**t)*period_cost

        # Check IL bounds
        if not dc_il_bounds[0] <= dc_il <= dc_il_bounds[1]:    # If DC IL outside allowed limit
            dc_il = max(min(dc_il, dc_il_bounds[1]), dc_il_bounds[0])
        if not w_il_bounds[0] <= w_il <= w_il_bounds[1]:       # If warehouse IL outside allowed limit
            w_il = max(min(w_il, w_il_bounds[1]), w_il_bounds[0])

        # Add ILs, costs, demands, actions to the list of histories
        states_visited.append((dc_il, w_il, ) + dc_arr_orders + w_arr_orders)
        costs.append(period_cost)
        demands.append(d)
        actions.append((dc_q, w_q))

        if 0 < period_cost*(gamma**t) < tol:  # Convergence stopping criterion for negligible cost increase
            print(f"Converged at {t} when starting with {ips0}")
            break

    return states_visited, ILs_pre_demand, IPs_pre_demand, costs, demands, actions, total_cost



def clean_decentralised_2ech_policy(policy: Dict, S: int, Q_max:int, dwoc: bool = False):
    '''
    Reformats a decentralised policy to include decisions of warehouse and DC for each IL tuple
    
    Inputs: 
        policy: Dictionary containing the warehouse order policy for each IL tuple at (DC, warehouse)
        S: order up to level followed by DC
        Q_max: maximum allowed order quantity for DC
        dwoc (boolean): if True, implement DWOC, otherwise exclude

    Outputs:
        full_policy: Dictionary containing (DC, warehouse) optimal order quantity as the value 
            for each (DC, warehouse) IL combination
    '''


    full_policy = {(dc_il, w_il): (min(relu(S - dc_il), Q_max, relu(w_il)) if dwoc else min(relu(S - dc_il), Q_max), w_q) for (dc_il, w_il), w_q in policy.items()}
    return full_policy



def simulate_2ech_replications(policy: Dict, nreps: int, num_periods: int, gamma: float, h: List, p: List, 
                               demand_distribution: Dict, tol: float, ils0_set: Set, lead_times: List, simulation_func: Callable, seed=1234):
    '''
    Executes nruns replications of simulations of a 2-echelon supply chain experiencing
    stochastic demand and implementing a given policy.
    
    Inputs: 
        policy: Dictionary describing DC and warehouse ordering policy for a given DC, warehouse inventory level combination
        nreps: number of replications of the simulation
        num_periods: length of each simulation
        gamma: discount factor for costs incurred during each simulation
        h: unit holding cost at DC and warehouse
        p: unit backlog penalty cost at DC and warehouse
        demand_distribution: dictionary describing probability distribution of customer demand
        ils0_set: set of initial inventory level tuples for each simulation
        seed: seed for reproducibility
    
    Outputs:
        simulation_costs: Dictionary describing the average total discounted costs across nreps replications
                of simulations using the given policy from each starting initial inventory level scenario
    '''

    ils_set = sorted(policy.keys() & ils0_set)    # Take intersection of given initial inventory level set and possible starting states under given policy system
    simulation_costs = {key: 0 for key in ils_set}   # Dictionary to store average total discounted cost for each initial IL

    for key in ils_set:    # for each initial IL scenario
        np.random.seed(seed)   # set seed
        discounted_costs = []  # List to store total discounted costs from each simulation
        for _ in range(nreps): # conduct nreps replications
            sim_results = simulation_func(num_periods, gamma, key, h, p,
                                            demand_distribution, lead_times, policy, tol)        # individual simulation of num_periods length
            discounted_costs.append(sim_results[-1])        # Add total discounted cost from performed simulation                            
        simulation_costs[key] = sum(discounted_costs)/nreps # Average total discounted costs across nreps simulation replications

    return simulation_costs


def make_cost_plot_dict(optimal_dict, lead_times, n_ech):
    IPs = sorted(set(calculate_ip(state, lead_times, n_ech) for state in optimal_dict)) 
    
    DC_cost = dict((ip_dc, set()) for (ip_dc, ip_w) in IPs)
    W_cost = dict((ip_w, set()) for (ip_dc, ip_w) in IPs)
    
    for state, cost in optimal_dict.items():
        ip_dc, ip_w = calculate_ip(state, lead_times, n_ech)

        DC_cost[ip_dc].add((ip_w, cost/1000))
        W_cost[ip_w].add((ip_dc, cost/1000))

    return W_cost, DC_cost
    

def two_cost_plot(VI_dict, simulation_dict, lead_times, n_ech, colour_by="W", title=None):    
    
    W_cost, DC_cost = make_cost_plot_dict(VI_dict, lead_times, n_ech)
    W_2_cost, DC_2_cost = make_cost_plot_dict(simulation_dict, lead_times, n_ech)

    cost_dict = W_cost if colour_by == "DC" else DC_cost
    cost_sim_dict = W_2_cost if colour_by == "DC" else DC_2_cost
    x_site = "DC" if colour_by=="W" else "warehouse"

    # Creates a cost vs IL level plot where each line represents the IL at the site in "colour_by"
    cmap = plt.get_cmap("tab20")
    keys = sorted(cost_sim_dict.keys())
    colors = cmap([i/(len(keys)-1) for i in range(len(keys))])

    for ip, color in zip(keys, colors):
        if ip in cost_dict:
            ips, costs = zip(*sorted(cost_dict[ip]))
            plt.plot(ips, costs, "--", label=ip, color=color)
        if ip in cost_sim_dict:   
            ips, costs = zip(*sorted(cost_sim_dict[ip])) 
            plt.plot(ips, costs, ":", color=color)

    plt.xlabel(f"Initial inventory position at {x_site}", fontsize=18)
    plt.ylabel("Optimal Cost (in 1000s)", fontsize=18)
    plt.title(f"DP (dashed) and simulation (dotted) costs in a {title}")
    if min(keys) >= 0:
        leg = plt.legend(title=f"{colour_by} IP", bbox_to_anchor=(1,1))
    else:
        leg = plt.legend(title=f"{colour_by} IP", bbox_to_anchor=(1,1), fontsize=10, ncol=1)
    for line in leg.get_lines():
        # line.set_linestyle((0, (3, 5, 1, 5)))
        line.set_linestyle('-')

    plt.grid()
    plt.tight_layout()
    # plt.savefig(f"Figures/multi_echelon/twocost_{cost_names[0]}_dash_{cost_names[1]}_dot_leg{colour_by}_cap{capacity}_MOQ{maxA}_sl{cb[0]*100/(cb[0]+h[0]):.1f}.pdf", dpi=300)
    plt.show() 


def cost_diff_plot(VI_dict, simulation_dict, lead_times, n_ech, colour_by = "W", title=None):
    W_cost, DC_cost = make_cost_plot_dict(VI_dict, lead_times, n_ech)
    W_sim_cost, DC_sim_cost = make_cost_plot_dict(simulation_dict, lead_times, n_ech)

    cost_dict = W_cost if colour_by == "DC" else DC_cost
    cost_sim_dict = W_sim_cost if colour_by == "DC" else DC_sim_cost
    x_site = "DC" if colour_by == "W" else "warehouse"

    # Creates a cost vs IL plot where each line represents the IP at the other site
    cmap = plt.get_cmap("tab20")
    keys = sorted(cost_sim_dict.keys() & cost_dict.keys())
    colors = cmap([i/(len(keys) - 1) for i in range(len(keys))])

    for ip, color in zip(keys, colors):
        x_ips = []
        cost_diff = []
        vi_ips_dict = dict(cost_dict[ip])
        sim_ips_dict = dict(cost_sim_dict[ip])
        common_ips = vi_ips_dict.keys() & sim_ips_dict.keys()

        for ipn in sorted(common_ips):
            x_ips.append(ipn)
            cost_diff.append(abs(vi_ips_dict[ipn] - sim_ips_dict[ipn]))

        plt.plot(x_ips, cost_diff, color=color, label=ip)

    plt.xlabel(f"Initial inventory position at {x_site}", fontsize=18)
    plt.ylabel(f"Difference in Cost (in 1000s)", fontsize=18)
    plt.title(f"Difference in DP and simulation costs in a {title}")
    if min(keys) >= 0:
        leg = plt.legend(title=f"{colour_by} IP", bbox_to_anchor=(1,1))
    else:
        leg = plt.legend(title=f"{colour_by} IP", bbox_to_anchor=(1,1), fontsize=10, ncol=1)
    for line in leg.get_lines():
        # line.set_linestyle((0, (3, 5, 1, 5)))
        line.set_linestyle('-')

    plt.grid()
    plt.tight_layout()
    # plt.savefig(f"Figures/multi_echelon/diffcost_{cost_names[0]}_dash_{cost_names[1]}_dot_leg{colour_by}_cap{capacity}_MOQ{maxA}_sl{cb[0]*100/(cb[0]+h[0]):.1f}.pdf", dpi=300)
    plt.show() 

                

            # vi_ips, vi_costs = zip(*sorted(cost_dict[ip]))
            # sim_ips, sim_costs = zip(*sorted(cost_sim_dict[ip]))



def calculate_ip(state: tuple, lead_times: List, n_ech: int):
    ''' Calculates the inventory position for each site in the supply chain'''
    ips = []
    next_lead_idx = n_ech
    for ech in range(n_ech): # for each site
        ips.append(state[ech] + sum(state[next_lead_idx:next_lead_idx+lead_times[ech]])) # add inventory level + outstanding orders for the current site
        next_lead_idx += lead_times[ech]

    return tuple(ips)


def calculate_echelon_ip(state: tuple, lead_times: List, n_ech: int):
    ''' Calculates the inventory position for each site in the supply chain'''
    ips = calculate_ip(state, lead_times, n_ech)
    ech_ips = [ips[0]]
    for ech in range(1, n_ech): # for each site
        ech_ips.append(ips[ech] + sum(ech_ips))

    return tuple(ech_ips)