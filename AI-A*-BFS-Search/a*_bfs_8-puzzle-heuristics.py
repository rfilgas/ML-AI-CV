import numpy as np
import queue
from dataclasses import dataclass, field
from typing import Any

# Ryan Filgas
# Intro to AI

# This program is designed to solve the 8-puzzle problem. In order to
# toggle settings, just switch the ASTAR and CURRENT_HEURISTIC variables
# below, as well as the solution if desired then run the program. The program will
# output all intermediate states on the most efficient path it found to the goal.

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
    depth: Any=field(compare=False)
    camefrom: Any=field(compare=False)


PATH_COST = 1 # Reducing this increases speed for worse results. Increasing does the opposite.
EMPTY = 0 # This will change what the empty charactar looks like. I kept it 0 to avoid mixed types.
# These specify the limits on the grid. They unfortunately don't work on larger grids at the moment.
MAX = 9 # The total number of integers in the goal state
WIDTH = 3 # The width of the goal state
SEARCH_LIMIT = 2000 # Upper limit on states searched.
# The algorithm checks if there is a possible solution, so one will be found
# if this is set to an arbitrarily large number, it may just take a long time.
TRANSITION = {"left": -1, "right": 1, "up": -WIDTH, "down": WIDTH}

# These set the heuristic to use below
HEURISTIC1 = "manhattanDistance"
HEURISTIC2 = "misplacedSquares"
HEURISTIC3 = "combo"

########COMMAND CENTER#######################################
# START = np.array([3,2,4,1,5,6,7,8,0])
START = None # If this is set to None, the start will be randomly generated.
GOAL = np.array([1,2,3,4,5,6,7,8,0]) # The state must be a numpy array like this.
# This will toggle using A* or basic Best First Search
ASTAR = False
CURRENT_HEURISTIC = HEURISTIC3
RUNS = 5



# This function tests if the goal state has been reached.
def goal_test(state, goal):
    return np.allclose(state, goal)



# This function takes in an action and transitions the state
# based on the action to a new state.
def transition_model(state, action):
    newstate = state.copy()
    current = np.where(state == EMPTY)[0][0]
    next = current + TRANSITION[action]
    if current in (2,5,8) and next == current+1:
        return None
    if current in (0,3,6) and next == current-1:
        return None
    if next < 0 or next >= MAX:
        return None
    else:
        newstate[current] = newstate[next]
        newstate[next] = EMPTY
        return newstate



def is_valid_state(state1, state2, width):
    parities1 = 0
    parities2 = 0
    max = width **2
    for i, x in enumerate(state1):
        temp = state1[(i+1):max]
        for nums in temp:
            if nums < x and x != 0 and nums != 0:
                parities1 += 1

    for j, y in enumerate(state2):
        temp = state2[(j+1):max]
        for nums in temp:
            if nums < y and y != 0 and nums != 0:
                parities2 += 1
    return (parities1%2) == (parities2%2)



# This function counts the number of squares not in the desired state.
def num_misplaced(state):
    zero = np.where(GOAL == EMPTY)[0][0]
    truths = (state == GOAL)
    num_false = np.bincount(truths)[0]
    if state[zero] != 0:
        num_false -= 1
    return num_false



def manhattan_dist_sum(state, width):
    distance = 0
    for i, x in enumerate(state):
        goal_index = np.where(GOAL == x)[0][0]
        distance += get_distance(width, i, goal_index)
    return distance



# This function generates the manhattan distance from a 1D array.
# There is likely a better way.
def get_distance(width, position, destination):
    if destination == 0:
        return 0
    spot, spot2 = position + 1, destination + 1
    column1, column2, row, row2 = 0,0,0,0
    if spot % width == 0:
        column1 = width
    else:
        column1 = spot % width
    if spot2 % width == 0:
        column2 = width
    else:
        column2 = spot2 % width
    if spot % 3 == 0:
        row = int(spot / width)
    else:
        row = int(spot / width) + 1
    if spot2 % 3 == 0:
        row2 = int(spot2 / width)
    else:
        row2 = int(spot2 / width) + 1
    a1 = row-1
    a2 = column1-1
    b1 = row2-1
    b2 = column2 - 1
    distance = abs(a1 - b1) + abs(a2 - b2)
    return distance



# This function generates a random state.
def get_random_state(width,goal):
    state = np.arange(width**2)
    np.random.shuffle(state)
    while is_valid_state(state,goal,width) == False:
        np.random.shuffle(state)
    return state



# This function gets all possible potential next states from an input state
# and returns them as a tuple with the heuristic cost and the new state.
def get_state_pairs_manhattan(state, width):
    statelist = []
    state1 = transition_model(state, "up")
    state2 = transition_model(state, "down")
    state3 = transition_model(state, "right")
    state4 = transition_model(state, "left")
    if state1 is not None:
        cost1 = manhattan_dist_sum(state1, width)
        statelist.append((cost1, state1))
    if state2 is not None:
        cost2 = manhattan_dist_sum(state2, width)
        statelist.append((cost2, state2))
    if state3 is not None:
        cost3 = manhattan_dist_sum(state3, width)
        statelist.append((cost3, state3))
    if state4 is not None:
        cost4 = manhattan_dist_sum(state4, width)
        statelist.append((cost4, state4))
    return statelist



# This function gets all possible potential next states from an input state
# and returns them as a tuple with the heuristic cost and the new state.
def get_state_pairs_combo(state, width):
    statelist = []
    state1 = transition_model(state, "up")
    state2 = transition_model(state, "down")
    state3 = transition_model(state, "right")
    state4 = transition_model(state, "left")
    if state1 is not None:
        cost1 = manhattan_dist_sum(state1, width) + num_misplaced(state1)
        statelist.append((cost1, state1))
    if state2 is not None:
        cost2 = manhattan_dist_sum(state2, width) + num_misplaced(state2)
        statelist.append((cost2, state2))
    if state3 is not None:
        cost3 = manhattan_dist_sum(state3, width) + num_misplaced(state3)
        statelist.append((cost3, state3))
    if state4 is not None:
        cost4 = manhattan_dist_sum(state4, width) + num_misplaced(state4)
        statelist.append((cost4, state4))
    return statelist



# This function gets all possible potential next states from an input state
# and returns them as a tuple with the heuristic cost and the new state.
def get_state_pairs_squares(state, width):
    statelist = []
    state1 = transition_model(state, "up")
    state2 = transition_model(state, "down")
    state3 = transition_model(state, "right")
    state4 = transition_model(state, "left")
    if state1 is not None:
        cost1 = num_misplaced(state1)
        statelist.append((cost1, state1))
    if state2 is not None:
        cost2 = num_misplaced(state2)
        statelist.append((cost2, state2))
    if state3 is not None:
        cost3 = num_misplaced(state3)
        statelist.append((cost3, state3))
    if state4 is not None:
        cost4 = num_misplaced(state4)
        statelist.append((cost4, state4))
    return statelist



### ASTAR  & BFS ALGORITHM ##############################
# The A* algorithm uses a priority queue and a travel cost + heuristic to determine which
# path to take next. The top of the queue is opened to unravel new stated which then get added
# onto the queue. Each layer of nodes adds a new penalty. This process repeats itself until the
# target state is found. The BFS or best first search algorithm ommits the path cost.
def search(WIDTH, state,goal, heuristic, astar):

    visited_index, steps, depth, max = 0, 0, 0, SEARCH_LIMIT
    temp_state = state.copy()
    self_node = PrioritizedItem(0,temp_state.copy(), depth, None)
    state_queue = queue.PriorityQueue()
    visited_nodes = [self_node]
    solution = []

    # if we're not in the goal state search
    while goal_test(temp_state, goal) == False:
        # Search all possible paths that don't go backwards or hit a wall.
        if heuristic == "manhattanDistance":
            paths = np.array(get_state_pairs_manhattan(temp_state, WIDTH), dtype=object)
        elif heuristic == "misplacedSquares":
            paths = np.array(get_state_pairs_squares(temp_state, WIDTH), dtype=object)
        elif heuristic == "combo":
            paths = np.array(get_state_pairs_combo(temp_state, WIDTH), dtype=object)

        # Add states to the queue if they haven't been visited already.
        for pair in paths:
            temp = list(pair[1])
            covered = False
            # This avoids a python syntax error that won't check a 1 item list.
            if len(visited_nodes) > 1:
                for node in visited_nodes:
                    if list(node.item) == temp:
                        covered = True
            # if we haven't opened this node add it to the list.
            if covered == False:
                if astar is True:
                    state_queue.put(PrioritizedItem(pair[0] + depth,pair[1], depth+PATH_COST, visited_index))
                else:
                    # A* is not being used in this case. Ignore depth when adding node.
                    state_queue.put(PrioritizedItem(pair[0],pair[1], depth+PATH_COST, visited_index))
        node = state_queue.get()
        temp_state = node.item
        depth = node.depth
        visited_nodes.append(node)
        visited_index += 1 # This ensures the path from the end can be retraced.

        # halting condition
        steps += 1
        if steps > max:
            print("Exited!")
            return None, None, None

    # retrace steps to put together the solution path
    current = len(visited_nodes) - 1
    temp = None
    while current != None:
        temp = visited_nodes[current]
        solution.append(temp)
        current = temp.camefrom
    solution.reverse()
    return temp_state, visited_nodes, solution
### ASTAR ALGORITHM ##############################



### BFS and A* TEST ###################################
steps = 0
count = 0
states_explored = 0
runs = RUNS

if START is not None and is_valid_state(START,GOAL,WIDTH) == False:
        print("INVALID STATE: TRY AGAIN")
else:
    with open('readme.txt', 'w') as f:
        if START is not None:
            runs = 1
        for x in range(runs):
            if START is None:
                state = get_random_state(WIDTH, GOAL)
            else:
                state = START
                if is_valid_state(START,GOAL,WIDTH) == False:
                    print("INVALID STATE: TRY AGAIN")

            new_state, visited_nodes, solution = search(WIDTH, state,GOAL, CURRENT_HEURISTIC, ASTAR) # The input heuristic is entered here.
            if new_state is None:
                f.write(str("Solution not found.\n"))
            else:
                f.write(str("SOLUTION_LENGTH: " + str(len(solution)) + "\n"))
                steps += len(solution)
                count += 1
                f.write(str("--------------------\n"))
                for node in solution:
                    f.write(str(node.item) + " --> ")
                states_explored += len(visited_nodes)
                f.write(str("\n--------------------\n"))
        f.write(str("Average number of states explored: " + str(states_explored/count)+ "\n"))
        f.write(str( "Average number of steps in found solution:" + str(steps/count)+ "\n"))
### BFS and A* TEST ###################################