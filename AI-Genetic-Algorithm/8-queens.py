import numpy as np
import functools
from functools import total_ordering
import bisect
import matplotlib.pyplot as plt

# Ryan Filgas
# AI Fall 2022

STATESIZE = 8
MAXFITNESS = 28

# Allow for sorting of individuals
@functools.total_ordering
class member:
    def __init__(self, fitness, position):
        self.fitness, self.position = fitness, position
    def __lt__(self, other):
        return (self.fitness) < (other.fitness)
    def __eq__(self, other):
        return (self.fitnessn) == (other.fitness)

    def copy(self):
        return member(self.fitness, self.position)



#Check pairs of queens that aren't attacking.
def fitness(queens):
    length = len(queens)
    num = 0
    count = 0

    for i in range(0, length):
        for j in range(i+1, length):
            # Horizontal
            if queens[i] == queens[j]:
                count += 1
            # Diagonal
            elif (j - i) == abs((queens[j] - queens[i])):
                count += 1
    return (MAXFITNESS - count)



# Generate a population for start.
def generate_pop(population):
    sorted_pool = list()
    initial_fitness = 0
    pop_generator = np.random.randint(1, high=(STATESIZE+1), size=(population,STATESIZE), dtype=int)
    for pop in pop_generator:
        my_fitness = fitness(pop)
        initial_fitness += my_fitness
        new_pop = member(my_fitness, pop)
        bisect.insort(sorted_pool, new_pop)
    return sorted_pool, initial_fitness



# create children using crossover and mutation
def generate_successors(child1, child2, MutationPct):
    max = len(child1.position)
    crossover = np.random.randint(1, high=max-1, dtype=int)
    new1 = np.array(list(child1.position[0:crossover]) + list(child2.position[crossover:max]))
    new2 = np.array(list(child2.position[0:crossover]) + list(child1.position[crossover:max]))

    # mutate children randomly
    if np.random.choice([0,1],1, p=(1-MutationPct, MutationPct))[0] == 1:
        mutate = np.random.randint(0, high=max, dtype=int)
        new1[mutate] = np.random.randint(0, high=max+1, dtype=int)
        if new1[mutate] <= 0:
            new1[mutate] = 1
        if new1[mutate] >= STATESIZE:
            new1[mutate] = STATESIZE

    if np.random.choice([0,1],1, p=(1-MutationPct, MutationPct))[0] == 1:
        mutate = np.random.randint(0, high=max, dtype=int)
        new2[mutate] = np.random.randint(0, high=max+1, dtype=int)
        if new2[mutate] <= 0:
            new2[mutate] = 1
        if new2[mutate] >= STATESIZE:
            new2[mutate] = STATESIZE

    # Create member objects with the new specimens
    mutated1 = member(fitness(new1), new1)
    mutated2 = member(fitness(new2), new2)
    return mutated1, mutated2

def group_fitness(population):
    sum = 0
    max = len(population)
    for i in population:
        sum += i.fitness
    return (sum/max), sum



def run_game(PopulationSize, NumIterations, MutationPct):
    sorted_pool, initial_fitness = generate_pop(PopulationSize)
    childStart = np.max(sorted_pool).copy()
    startingFitness = initial_fitness/PopulationSize
    average_fitness = []
    top2List = []

    for i in range(NumIterations):

        new_pop = []
        avg_fitness, total_fitness = group_fitness(sorted_pool)
        average_fitness.append(avg_fitness)
        # Get probabilities and select parents
        probabilities = [(k.fitness/total_fitness) for k in sorted_pool]

        ##################
        pop_control = int(PopulationSize/2)
        for j in range(pop_control):
            # Set to arbitrarily large number so loop begins.
            parent1_idx, parent2_idx = 999999999, 999999999

            # Assign indexes based on distribution. They must not choose the same individual.
            # As the array will be one smaller after the first index is popped off, we have to check
            # that the next thing popped off doesn;t fall offf the array.
            while parent2_idx == parent1_idx or parent2_idx >= PopulationSize-1 or parent1_idx >= PopulationSize:
                parent1_idx = int(np.random.choice(PopulationSize, 1, replace=False, p=probabilities))
                parent2_idx = int(np.random.choice(PopulationSize, 1, replace=False, p=probabilities))
            # Retrieve parents
            parent1 = sorted_pool[parent1_idx]
            parent2 = sorted_pool[parent2_idx]

            #generate children
            child_a, child_b = generate_successors(parent1, parent2, MutationPct)
            new_pop.append(child_a)
            new_pop.append(child_b)
        sorted_pool = new_pop
        new_pop = []
        top2List.append(np.max(sorted_pool))

    childEnd = np.max(sorted_pool).fitness
    finalRound = NumIterations
    lastChild = np.max(sorted_pool)
    endingFitness = average_fitness[len(average_fitness)-1]

    return startingFitness, endingFitness, childStart, childEnd, finalRound, average_fitness, top2List, lastChild


PopulationSize = 10
NumIterations= 100
MutationPct = .05

startingFitness, endingFitness, childStart, childEnd, finalRound, averageFitness, top2List, lastChild = run_game(PopulationSize, NumIterations, MutationPct)

print("\n")
print("Starting Fitness: ", startingFitness)
print("Ending Fitness: ", endingFitness)
print("Fitness of best start: ", childStart.fitness)
print("Fitness of best end: ", childEnd)
print("Final Round Was: ", len(averageFitness))
print("\n\n")

# Gather samples
check = [top2List[int(len(top2List)/i)-1] for i in range(1,9)]

#Print the first best, the middle 80% and the final best.
print(list(top2List[0].position))
print([list(i.position) for i in check])
print(list(lastChild.position))

plt.plot(np.arange(len(averageFitness)),averageFitness)
plt.title('Average Fitness (Goal: 28) \nPopulation: ' + str(PopulationSize) + '\nMutation Pct: ' + str(MutationPct))
plt.xlabel('Iterations')
plt.ylabel('Non-Attacking Queen Pairs')
plt.show()