from collections import defaultdict
import traceback


class MDP():
    """Class for representing a Gridworld MDP.

    States are represented as (x, y) tuples, starting at (1, 1).  It is assumed that there are
    four actions from each state (up, down, left, right), and that moving into a wall results in
    no change of state.  The transition model is specified by the arguments to the constructor (with
    probability prob_forw, the agent moves in the intended direction. It veers to either side with
    probability of (1-prob_forw)/2 each.  If the agent runs into a wall, it stays in place.
    """

    def __init__(self, num_rows, num_cols, rewards, terminals, prob_forw, reward_default=0.0):
        """
        Constructor for this MDP.

        Args:
            num_rows: the number of rows in the grid
            num_cols: the number of columns in the grid
            rewards: a dictionary specifying the reward function, with (x, y) state tuples as keys,
                and rewards amounts as values.  If states are not specified, their reward is assumed
                to be equal to the reward_default defined below
            terminals: a list of state (x, y) tuples specifying which states are terminal
            prob_forw: probability of going in the intended direction
            reward_default: reward for any state not specified in rewards
        """
        self.nrows = num_rows
        self.ncols = num_cols
        self.states = []
        for i in range(num_cols):
            for j in range(num_rows):
                self.states.append((i+1, j+1))
        self.rewards = rewards
        self.terminals = terminals
        self.prob_forw = prob_forw
        self.prob_side = (1.0 - prob_forw)/2
        self.reward_def = reward_default
        self.actions = ['up', 'right', 'down', 'left']

    def get_states(self):
        """Return a list of all states as (x, y) tuples."""
        return self.states

    def get_actions(self, state):
        """Return list of possible actions from each state."""
        return self.actions

    def get_successor_probs(self, state, action):
        """Returns a dictionary mapping possible successor states to their transition probabilities
        for the given state and action.
        """
        if self.is_terminal(state):
            return {}  # we cant move from terminal state since we end

        x, y = state
        succ_up = (x, min(self.nrows, y+1))
        succ_right = (min(self.ncols, x+1), y)
        succ_down = (x, max(1, y-1))
        succ_left = (max(1, x-1), y)

        succ__prob = defaultdict(float)
        if action == 'up':
            succ__prob[succ_up] = self.prob_forw
            succ__prob[succ_right] += self.prob_side
            succ__prob[succ_left] += self.prob_side
        elif action == 'right':
            succ__prob[succ_right] = self.prob_forw
            succ__prob[succ_up] += self.prob_side
            succ__prob[succ_down] += self.prob_side
        elif action == 'down':
            succ__prob[succ_down] = self.prob_forw
            succ__prob[succ_right] += self.prob_side
            succ__prob[succ_left] += self.prob_side
        elif action == 'left':
            succ__prob[succ_left] = self.prob_forw
            succ__prob[succ_up] += self.prob_side
            succ__prob[succ_down] += self.prob_side
        return succ__prob

    def get_reward(self, state):
        """Get the reward for the state, return default if not specified in the constructor."""
        return self.rewards.get(state, self.reward_def)

    def is_terminal(self, state):
        """Returns True if the given state is a terminal state."""
        return state in self.terminals


def value_iteration(mdp, gamma, epsilon):
    """Calculate the utilities for the states of an MDP.

    Args:
        mdp: An instance of the MDP class defined above, describing the environment
        gamma: the discount factor
        epsilon: the change threshold to use when determining convergence.  The function returns
            when none of the states have a utility whose change from the previous iteration is more
            than epsilon

    Returns:
        A python dictionary, with state (x, y) tuples as keys, and converged utilities as values.
    """
    utilities = {}
    updated_utilities = {}
    #set the minReward as the greatest value
    min_reward = float("inf")
    #for each state:
    for s in mdp.get_states():
        utilities[s] = mdp.get_reward(s)
        if utilities[s] < min_reward:
            #update minReward
            min_reward = utilities[s]
    convergence = False

    while not convergence:
        convergence_val = 0
        #for each state:
        for s in mdp.get_states():
            temp = utilities[s]
            max_val = min_reward
            #for each action:
            for action in mdp.get_actions(s):
                #compare the values and update
                max_val = max(max_val, utility_prob_sum(
                    mdp.get_successor_probs(s, action), utilities))
            #apply the formula
            updated_utilities[s] = (mdp.get_reward(s) + gamma * max_val)
            #compare the values and update
            convergence_val = max(convergence_val, abs(
                temp - updated_utilities[s]))
        #when iteration is done for all states, update the states' original values
        for state in updated_utilities.keys():
            utilities[state] = updated_utilities[state]
        #check convergence
        if convergence_val <= epsilon:
            convergence = True
    return updated_utilities


def utility_prob_sum(prob_state_dict, utility_dict):
    sum_val = 0
    for suc, prob in prob_state_dict.items():
        sum_val += prob * utility_dict[suc]
    return sum_val


def derive_policy(mdp, utility):
    """Create a policy from an MDP and a set of utilities for each state.

    Args:
        mdp: An instance of the MDP class defined above, describing the environment
        utility: A dictionary mapping state (x, y) tuples to a utility value (perhaps calculated
            from value iteration)

    Returns:
        policy: A dictionary mapping state (x, y) tuples to the optimal action for that state (one
            of 'up', 'down', 'left', 'right', or None for terminal states)
    """
    policy = {}
    #for each state:
    for s in mdp.get_states():
        #check if the state is terminal
        if mdp.is_terminal(s):
            policy[s] = None
        else:
            best_action = None
            #assign the lowest value as best_utility
            best_utility = float('-inf')
            #for each action:
            for a in mdp.get_actions(s):
                #calculate the expected utility(exp_util)
                exp_util = sum([prob*utility[s_p] for s_p,
                               prob in mdp.get_successor_probs(s, a).items()])
                #compare values and update
                if exp_util > best_utility:
                    best_utility = exp_util
                    best_action = a
            #update the policy of the state with the best action
            policy[s] = best_action
    return policy


def utils_and_policy(emdeepee, gamma, epsilon):
    """Calculate the utilities for the states of an MDP and create a policy from
    an MDP and a set of utilities for each state.

    Args:
        emdeepee: An instance of the MDP class defined above, describing the environment
        gamma: the discount factor
        epsilon: the change threshold to use when determining convergence.  The function returns
            when none of the states have a utility whose change from the previous iteration is more
            than epsilon

    Returns:
        utility: A dictionary mapping state (x, y) tuples to a utility value (perhaps calculated
            from value iteration)
        policy: A dictionary mapping state (x, y) tuples to the optimal action for that state (one
            of 'up', 'down', 'left', 'right', or None for terminal states)
    """
    if emdeepee == None or gamma == None or epsilon == None:
        print("At least one of emdeepee, gamma, or epsilon was None in utils_and_policy.")
        return None, None
    utilities = value_iteration(emdeepee, gamma, epsilon)
    try:
        print(ascii_grid_utils(utilities))
        print()
        policy = derive_policy(emdeepee, utilities)
        print(ascii_grid_policy(policy))
    except:
        if utilities == None:
            print("Your value iteration returns None in gen_results.")
        else:
            print("Error Traceback:")
            print('―' * 10)
            print(traceback.format_exc(), '―' * 10)
            print("\nYour value iteration is likely returning the wrong format.")
        return None, None

    return utilities, policy


def ascii_grid_utils(utility):
    """Return an ascii-art gridworld with utilities.

    Args:
        utility: A dictionary mapping state (x, y) tuples to a utility value
    """
    return ascii_grid(dict([(k, "{:8.4f}".format(v)) for k, v in utility.items()]))


def ascii_grid_policy(actions):
    """Return an ascii-art gridworld with actions.

    Args:
        actions: A dictionary mapping state (x, y) tuples to an action (up, down, left, right)
    """
    symbols = {'up': '^^^', 'right': '>>>',
               'down': 'vvv', 'left': '<<<', None: ' x '}
    return ascii_grid(dict([(k, "   " + symbols[v] + "  ") for k, v in actions.items()]))


def ascii_grid(vals):
    """High-tech helper function for printing out values associated with a 2x3 MDP."""
    s = ""
    s += " ________________________________  \n"
    s += "|          |          |          | \n"
    s += "| {} | {} | {} | \n".format(vals[(1, 2)], vals[(2, 2)], vals[(3, 2)])
    s += "|__________|__________|__________| \n"
    s += "|          |          |          | \n"
    s += "| {} | {} | {} | \n".format(vals[(1, 1)], vals[(2, 1)], vals[(3, 1)])
    s += "|__________|__________|__________| \n"
    return s

