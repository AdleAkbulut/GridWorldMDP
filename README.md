# GridWorldMDP
An MDP simulator that drives the best policy and the utilities related to the best policy on a 2x3 grid.

This repository holds the files that contains the simulator and the tests. 

## mdp.py
This is the file that has the simulator.
Creates an MDP class and calculates the utilities of each state with value iteration algorithm. 
Drives the best policy for the grid. 
To activate all these functionalities utils_and_policy() method needs to be envoked.

## mdp_test.py
This is the file that has 4 different grids for testing.
Creates 4 different grids with different rewards and discount factors.
Prints the original grid and the updated grid with the best policy for each test.
