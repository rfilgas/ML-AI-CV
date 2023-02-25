import numpy as np
import matplotlib.pyplot as plt

# Ryan Filgas
# Reinforcement Learning
# Program 1 - Homework 2

""" Assignment Parameters:
Implement a 5 armed bandit problem with greedy and e-greedy action selection algorithms.
Compare the results of e-greedy action selection method (e=0.4) with the greedy one. Which
one works better over 100 time-steps in 200 runs? You can choose any distribution/values
for your reward function and/or other parameters.
"""

# The agent class controls the agents actions and perspective
# of the problem given feedback from the testbed.
class agent(object):
    # set up the state of the agent
    def __init__(self, arms, epsilon, estRewards=[]):
        self.arms = arms
        self.epsilon = epsilon
        self.estRewards = np.zeros(arms)
        self.numSelected = np.zeros(arms)

    def action(self):
        max = np.random.choice(np.where(self.estRewards == self.estRewards.max())[0])
        not_max = ([x for x in range(self.arms)])
        not_max.remove(max)
        explore = np.random.choice(not_max)

        if self.epsilon == 0:
            return max
        else:
            # epsilon % of the time: make a random choice
            if np.random.choice([0,1],1, p=(1-self.epsilon, self.epsilon))[0] == 1:
                return explore
            else: return max

    # given an action, update the internal state
    def updateRewards(self, reward, pulledArm):
        currentReward = self.estRewards[pulledArm]
        self.numSelected[pulledArm] += 1
        k = self.numSelected[pulledArm]
        self.estRewards[pulledArm] = currentReward + (1/k)*(reward - currentReward)
        return

    def reset(self):
        self.numSelected = np.zeros(self.arms)
        self.estRewards = np.zeros(self.arms)
        pass



# set up the environment to test in
class testbed(object):
    def __init__(self, arms, stdev, mean):
        self.arms = arms
        self.mean = mean
        self.stdev = stdev
        self.rewards = np.random.normal(self.mean, self.stdev, (self.arms))
        self.optReward = np.argmax(self.rewards)

    def determineReward(self, arm):
        earned = self.rewards[arm]
        # sample from unit distribution with the reward centered at the mean
        earned = np.random.choice(np.random.normal(earned, self.stdev, 1))
        return earned, self.optReward

    def reset(self):
        self.rewards = np.random.normal(self.mean, self.stdev, (self.arms))
        self.optReward = np.argmax(self.rewards)



# The game class takes care of a single game given the number of timesteps.
class game(object):

    def __init__(self, arms,timeSteps, epsilon, testbed, agent):
        self.arms = arms
        self.timesteps = timeSteps
        self.epsilon = epsilon
        self.testbed = testbed
        self.agent = agent
        self.scores = []
        self.optScores = []

    def play(self):
        for x in range(self.timesteps):
            # evaluate state and choose action
            pulled_arm = self.agent.action()
            # take the action
            reward, optReward = self.testbed.determineReward(pulled_arm)
            # update state
            self.agent.updateRewards(reward, pulled_arm)
            # Add to reward lists
            self.scores.append(reward)
            self.optScores.append(optReward)
        return self.scores, self.optScores

    def reset(self):
        self.scores = []
        self.optScores = []

# Test
ARMS = 5
TIMESTEPS = 100
RUNS = 200
epsilon1 = 0 #Epsilon determines how often the agent chooses to deviate from a greedy action
STDEV = 1
MEAN = 0

sumRewards = np.zeros(TIMESTEPS)
sumOptimal = np.zeros(TIMESTEPS)
rewardsList = []
sumList = []
averageList = list()
epsilonList = [0,.01,.1,.4] #Add all epsilons to test

for test in epsilonList:
    epsilon1 = test
    for x in range(RUNS):
        # set up game
        myTestbed = testbed(arms = ARMS, stdev = STDEV, mean = MEAN)
        myAgent = agent(arms=ARMS, epsilon=epsilon1)
        myGame = game(timeSteps=TIMESTEPS,epsilon=epsilon1, arms=ARMS, testbed=myTestbed, agent=myAgent)

        # need both scores and optimal scores to benchmark how well the algorithm works.
        scores, optimalScores = myGame.play()
        sumRewards  = sumRewards + scores
        sumOptimal = sumOptimal + optimalScores

        # reset for next run
        myAgent.reset()
        myTestbed.reset()
        myGame.reset()
    average = sumRewards/sumOptimal
    averageList.append(average)
    sumRewards = np.zeros(TIMESTEPS)
    sumOptimal = np.zeros(TIMESTEPS)

# Plot results
t = np.array([x for x in range(TIMESTEPS)]) + 1
for i, chart in enumerate(averageList):
    label = "Epsilon: " + str(epsilonList[i])
    plt.plot(t,chart, label=label)
plt.title('Average performance per step over 200 runs')
plt.xlabel('Timesteps')
plt.ylabel('Percent of optimal solution')
plt.legend(loc='lower right')
plt.show()