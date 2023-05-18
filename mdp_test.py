import mdp

def test1():
    gridworld = mdp.MDP(2, 3,
                        rewards={(2, 2): -10, (3, 2): 20, (1, 2): -1,
                                 (3, 1): -1, (2, 1): -1, (1, 1): -1},
                        terminals=[(3, 2)],
                        prob_forw=0.8)
    epsilon = 0.01
    discount_factor = 0.8

    print("\n", '─' * 50, "\n", "Test 1 Start State")
    print(mdp.ascii_grid_utils(gridworld.rewards))
    print("\n", "Rewards With Best Policy")
    utilities, policy = mdp.utils_and_policy(
        gridworld, discount_factor, epsilon)

    return {"gridworld": gridworld, "epsilon": epsilon,
            "discount_factor": discount_factor, "utilities": utilities,
            "policy": policy}


def test2():
    gridworld = mdp.MDP(2, 3,
                        rewards={(2, 2): -100, (3, 2): 20, (1, 2): -1,
                                 (3, 1): -1, (2, 1): -1, (1, 1): -1},
                        terminals=[(3, 2)],
                        prob_forw=0.8)
    epsilon = 0.01
    discount_factor = 0.8

    print("\n", '─' * 50, "\n", "Test 2 Start State")
    print(mdp.ascii_grid_utils(gridworld.rewards))
    print("\n", "Rewards With Best Policy")
    utilities, policy = mdp.utils_and_policy(
        gridworld, discount_factor, epsilon)

    return {"gridworld": gridworld, "epsilon": epsilon,
            "discount_factor": discount_factor, "utilities": utilities,
            "policy": policy}


def test3():
    gridworld = mdp.MDP(2, 3,
                        rewards={(2, 2): -10, (3, 2): 20, (1, 2): -1,
                                 (3, 1): -1, (2, 1): -1, (1, 1): -1},
                        terminals=[(3, 2)],
                        prob_forw=0.5)
    epsilon = 0.01
    discount_factor = 0.8

    print("\n", '─' * 50, "\n", "Test 3 Start State")
    print(mdp.ascii_grid_utils(gridworld.rewards))
    print("\n", "Rewards With Best Policy")
    utilities, policy = mdp.utils_and_policy(
        gridworld, discount_factor, epsilon)

    return {"gridworld": gridworld, "epsilon": epsilon,
            "discount_factor": discount_factor, "utilities": utilities,
            "policy": policy}


def test4():
    gridworld = mdp.MDP(2, 3,
                        rewards={(2, 2): -10, (3, 2): 20, (1, 2): -1,
                                 (3, 1): -1, (2, 1): -1, (1, 1): -1},
                        terminals=[(3, 2)],
                        prob_forw=0.8)
    epsilon = 0.01
    discount_factor = 0.6

    print("\n", '─' * 50, "\n", "Test 4 Start State")
    print(mdp.ascii_grid_utils(gridworld.rewards))
    print("\n", "Rewards With Best Policy")
    utilities, policy = mdp.utils_and_policy(
        gridworld, discount_factor, epsilon)

    return {"gridworld": gridworld, "epsilon": epsilon,
            "discount_factor": discount_factor, "utilities": utilities,
            "policy": policy}


##########################

if __name__ == "__main__":
    #test different gridworls
    test1_results = test1()
    test2_results = test2()
    test3_results = test3()
    test4_results = test4()
