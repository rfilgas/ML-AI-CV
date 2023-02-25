import numpy as np
import matplotlib.pyplot as plt

# Ryan Filgas
# AI
# Program 3

class qnode:
    def __init__(self,):
        self.north = 0
        self.east = 0
        self.south = 0
        self.west = 0
        self.current = 0

    def set_q(self, index, value):
        if index == 0:
            self.north = value
        elif index == 1:
            self.east = value
        elif index == 2:
            self.south = value
        elif index == 3:
            self.right = value
        elif index == 4:
            self.current = value
        else: return -1

    def epsilon_max(self, epsilon, is_repeat):
        x = np.array([self.north, self.east, self.south, self.west, self.current])
        max = np.random.choice(np.where(x == x.max())[0]) #get index for max val
        not_max = ([x for x in range(len(x))]) #indexes of x
        not_max.remove(max) #removes index option from choices
        not_max = np.array(not_max)
        if is_repeat:
            max2 = max
            while max2 == max:
                max2 = np.random.choice(np.where(x == sorted(x, reverse=True)[1])[0])
            not_max = ([x for x in range(len(x))])
            not_max.remove(max)
            not_max.remove(max2)
            max = max2
        explore = np.random.choice(not_max)
        if epsilon == 0:
            return max, x[max]
        else:
            # epsilon % of the time: make a random choice
            choice = np.random.choice([0,1],1, p=(1-epsilon, epsilon))[0]
            if choice == 1:
                return explore, x[explore]
            else: return max, x[max]

    def true_max(self):
        x = np.array([self.north, self.east, self.south, self.west, self.current])
        return max(x)


class Robot:
    def __init__(self, environment, episodes, actions, epsilon, \
        learning_rate, discount, epsilon_decay):
        self.environment = environment
        self.position = environment.check_position()
        self.episodes = episodes
        self.actions = actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.learning_rate = learning_rate
        self.q_matrix = np.array([qnode() for x in range(self.environment.size[0] \
            * self.environment.size[1])]).reshape(self.environment.size)

    def make_matrix(self):
        return np.zeros(self.environment.size)

    def action_map(self, input):
        if input == 0:
            return self.environment.go_north(),
        elif input == 1:
            return self.environment.go_east(),
        elif input == 2:
            return self.environment.go_south(),
        elif input == 3:
            return self.environment.go_west(),
        elif input == 4:
            return self.environment.pick_up_can()

    def update_position(self):
        self.position = self.environment.check_position()

    def get_q_vals(self):
        return self.north, self.east, self.south, self.west, self.current

    def run(self):
        episode_rewards = []
        all_episode_rewards = []
        for i in range(self.episodes):
            ep_reward = float(0)
            if i % 50 == 0:
                self.epsilon = self.epsilon * self.epsilon_decay
            self.environment.reset()
            prev_position = None
            for t in range(self.actions):
                self.update_position()
                Q = self.q_matrix[self.position[0]][self.position[1]]

                repeat = False
                if self.is_repeat(prev_position, self.position):
                    repeat = True

                index, max = Q.epsilon_max(self.epsilon, repeat)
                given_reward = self.environment.move(index)

                # Add reward to accumulator
                ep_reward += given_reward
                prev_position = self.position
                self.update_position()

                # Get Q' vals
                next_Q = self.q_matrix[self.position[0]][self.position[1]]


                next_index, next_max = next_Q.epsilon_max(self.epsilon, False)


                # update Q value
                Q_update = max + self.learning_rate * (given_reward + (self.discount * next_max) - max)
                Q.set_q(index, Q_update)
            if i % 100 == 0: # keep t
                episode_rewards.append(ep_reward)
                print("Episode: ", i)
            all_episode_rewards.append(ep_reward)
        self.display_qmax()
        return episode_rewards, all_episode_rewards

    def display_qmax(self,):
        z = np.array([0 for x in range(self.environment.size[0] \
            * self.environment.size[1])]).reshape(self.environment.size)
        for x in range(self.environment.size[0]):
            for y in range(self.environment.size[1]):
                z[x][y] = self.q_matrix[x][x].true_max()
        print(z)

    def is_repeat(self, prev_pos, current_pos):
        if prev_pos is None or current_pos is None:
            return False
        if prev_pos[0] == current_pos[0] and prev_pos[1] == current_pos[1]:
            return True
        return False





class Environment:
    def __init__(self, size, movement_cost):
        self.size = size
        self.state = self.new_state()
        self.position = self.initial_position()
        self.out_of_bounds = -5 + movement_cost
        self.found_can = 10 + movement_cost
        self.empty_square = 0
        self.nothing_there = -1
        self.movement_cost = movement_cost
        self.reward_map = {
            -1: self.out_of_bounds,
            1: self.found_can,
            0: self.empty_square
        }

    def move(self, index):
        if index == 0:
            return self.go_north()
        elif index == 1:
            return self.go_east()
        elif index == 2:
            return self.go_south()
        elif index == 3:
            return self.go_west()
        elif index == 4:
            return self.pick_up_can()

    def reset(self):
        self.position = self.initial_position()
        self.state = self.new_state()

    def initial_position(self):
        return (np.random.choice(np.arange(self.size[0])), \
            np.random.choice(np.arange(self.size[0])))

    def new_state(self):
        return np.random.choice([0, 1], p=[.5, .5], size=self.size)

    def check_position(self):
        return self.position

    # Return true if out of bounds, false otherwise
    def is_out_of_bounds(self, position: tuple) -> bool:
        return (
            position[0] < 0) or (
            position[0] >= self.size[0]) or (
            position[1] < 0) or (
                position[1] >= self.size[1])

    def square_full(self, position):
        if not self.is_out_of_bounds(position):
            return self.state[position[0]][position[1]]
        else:
            return self.nothing_there

    ###### CHECKS ######
    def check_north(self) -> int:
        next_position = ((self.position[0] - 1), self.position[1])
        square_full = self.square_full(next_position)
        return self.reward_map[square_full]

    def check_south(self) -> int:
        next_position = ((self.position[0] + 1), self.position[1])
        square_full = self.square_full(next_position)
        return self.reward_map[square_full]

    def check_east(self) -> int:
        next_position = (self.position[0], (self.position[1] + 1))
        square_full = self.square_full(next_position)
        return self.reward_map[square_full]

    def check_west(self) -> int:
        next_position = (self.position[0], (self.position[1] - 1))
        square_full = self.square_full(next_position)
        return self.reward_map[square_full]

    def check_current(self) -> int:
        square_full = self.square_full(self.position)
        return self.reward_map[square_full]
    ###### CHECKS ######

    ###### ACTIONS ######
    def go_north(self) -> int:
        reward = self.check_north()
        if reward != self.out_of_bounds:
            self.position = (self.position[0] - 1, self.position[1])
            reward = self.movement_cost
        return reward

    def go_south(self) -> int:
        reward = self.check_south()
        if reward != self.out_of_bounds:
            self.position = (self.position[0] + 1, self.position[1])
            reward = self.movement_cost
        return reward

    def go_east(self) -> int:
        reward = self.check_east()
        if reward != self.out_of_bounds:
            self.position = (self.position[0], self.position[1] + 1)
            reward = self.movement_cost
        return reward

    def go_west(self) -> int:
        reward = self.check_west()
        if reward != self.out_of_bounds:
            self.position = (self.position[0], self.position[1] - 1)
            reward = self.movement_cost
        return reward

    def pick_up_can(self) -> int:
        reward = self.check_current()
        if reward == self.found_can:
            self.state[self.position[0]][self.position[1]] = self.empty_square
            return reward
        return self.nothing_there

    def display_position(self):
        print("Position:", self.position)

    def display_environment(self):
        x = self.state.copy()
        x[self.position[0]][self.position[1]] = 7
        print(x)

    ###### ACTIONS ######



# Initial settings requested
# N = 5,000 ; M = 200 ; 𝜂 = 0.2; 𝛾 = 0.9
episodes = 5000 # N Episodes
actions = 200  # M Steps
epsilon = .1
epsilon_decay = .9
learning_rate = .2
discount = .9
movement_cost = -.5
environment = Environment((10, 10), movement_cost)
robbie = Robot(environment, episodes, actions, epsilon, \
    learning_rate, discount, epsilon_decay)
# robbie.display_values()
episode_rewards, all_ep_rewards = robbie.run()

#Second run
robbie.epsilon = .1
robbie.epsilon_decay = 1
robbie.learning_rate = .2
test_rewards, all_test_rewards = robbie.run()
test_avg = np.average(all_test_rewards)
test_std = np.std(all_test_rewards)
print(test_avg)
print(test_std)
# print(episode_rewards)

y_axis = np.array(episode_rewards)
x_axis = np.arange(len(y_axis))
plt.plot(x_axis, y_axis, label = "Train")
plt.plot(x_axis, np.array(test_rewards), label = "Test")

# plt.plot(x_axis, y_axis)
plt.title('Q-learning Agent\n' + "Test Std: " \
    + str(test_std) + "\n" + "Test Avg: " + str(test_avg))
plt.xlabel('Episodes * 100')
plt.ylabel('Training Reward')
plt.legend()
plt.show()