import numpy as np
import matplotlib.pyplot as plt
import copy

# items
#   list of A [weight, value]

# individual, using stack encoding
#   list, X, containing A elements, each an integer [0, 10*A]

# mutation
#   any mutations are valid

# crossover
#   must keep the list the same size

# fitness
#   computed by looping through X, and picking the item: Xi % len(A), removing that element of A after selecting it
#   try tournament, all should be valid


def mutate_individual(individual):
    A = np.shape(individual)[0]
    index = np.random.randint(0, A)
    value = np.random.randint(0, 10*A)
    individual[index] = value

    return individual


def mutate_population(individuals):
    num_pop = np.shape(individuals)[0]
    for i in range(num_pop):
        individuals[i] = mutate_individual(individuals[i])

    return individuals


def generate_child(parent_1, parent_2):
    A = np.shape(parent_1)[0]
    index = np.random.randint(0, A)
    child = np.copy(parent_2)
    child[0:A] = parent_1[0:A]

    return child


def generate_children(parents):
    if len(parents) % 2 != 0:
        print('Number of parents not divisible by 2, killing program')
        quit()

    remaining_parents = list(copy.deepcopy(parents))
    children = []

    for i in range(len(parents) // 2):
        index_1 = np.random.randint(0, len(remaining_parents))
        remaining_parents.pop(index_1)
        index_2 = np.random.randint(0, len(remaining_parents))
        remaining_parents.pop(index_2)
        children.append(generate_child(parents[index_1], parents[index_2]))

    return children


def fitness(indiv_1, items, max_weight):
    item_indices = list(range(0, np.shape(items)[0]))
    weight = 0
    cost = 0
    i = 0
    for i in range(np.shape(items)[0]):
        indices_index = int(indiv_1[i] % len(item_indices))
        item = items[item_indices[indices_index]]
        item_indices.pop(indices_index)
        if weight + item[0] <= max_weight:
            weight += item[0]
            cost += item[1]
        else:
            return cost

    # Sum of weights < max_weight, end program
    quit()


def fitness_population(individuals, items, max_weight):
    fitness_arr = np.zeros(np.shape(individuals)[0])
    for i in range(np.shape(individuals)[0]):
        fitness_arr[i] = fitness(individuals[i], items, max_weight)

    return fitness_arr


def tournament(indices, fitness):
    if len(indices) % 4 != 0:
        print('Number of individuals not divisible by 4, killing program')
        quit()

    candidates = copy.deepcopy(indices)
    survivors = []

    for i in range(len(indices) // 2):
        index_1 = np.random.randint(0, len(candidates))
        candidates.pop(index_1)
        index_2 = np.random.randint(0, len(candidates))
        candidates.pop(index_2)
        if fitness[index_1] > fitness[index_2]:
            survivors.append(index_1)
        else:
            survivors.append(index_2)

    return survivors


# Possibilities:
# * delete half, generate 2 rounds of children
#   delete 1/3, use 1 recombination to refill
#   delete half, reproduce to 1.5x, delete all remaining parents
def selection(individuals, items, max_weight):

    fitness_arr = fitness_population(individuals, items, max_weight)

    individuals_indices = list(range(np.shape(individuals)[0]))
    parent_indices = tournament(individuals_indices, fitness_arr)
    parents = individuals[parent_indices]
    children_group_1 = generate_children(parents)
    children_group_2 = generate_children(parents)

    final_group = np.concatenate((parents, children_group_1, children_group_2))
    return final_group


def knapsack_evolution(individuals, items, max_weight, num_steps=1000):
    mean = np.zeros(num_steps)
    median = np.zeros(num_steps)
    std = np.zeros(num_steps)
    for step in range(num_steps):
        individuals = mutate_population(individuals)
        individuals = selection(individuals, items, max_weight)
        fitness_arr = fitness_population(individuals, items, max_weight)
        mean[step] = np.mean(fitness_arr)
        median[step] = np.median(fitness_arr)
        std[step] = np.std(fitness_arr)

    plot_vs_steps(mean, median, std)
    return individuals


def selection_asexual(individuals, items, max_weight):

    fitness_arr = fitness_population(individuals, items, max_weight)

    individuals_indices = list(range(np.shape(individuals)[0]))
    parent_indices = tournament(individuals_indices, fitness_arr)
    parents = individuals[parent_indices]
    parents_old = copy.deepcopy(parents)
    children = mutate_population(parents)

    final_group = np.concatenate((parents_old, children))
    return final_group


def knapsack_evolution_asexual(individuals, items, max_weight, num_steps=1000):
    mean = np.zeros(num_steps)
    median = np.zeros(num_steps)
    std = np.zeros(num_steps)
    for step in range(num_steps):
        individuals = mutate_population(individuals)
        individuals = selection_asexual(individuals, items, max_weight)
        fitness_arr = fitness_population(individuals, items, max_weight)
        mean[step] = np.mean(fitness_arr)
        median[step] = np.median(fitness_arr)
        std[step] = np.std(fitness_arr)

    plot_vs_steps(mean, median, std)
    return individuals


def plot_results(fitness_arr):
    plt.figure()
    x = np.arange(np.shape(fitness_arr)[0])
    plt.plot(x, fitness_arr, 'k.', lw=5)
    plt.xlabel('individual')
    plt.ylabel('cost')
    plt.title('Cost vs. individual')
    plt.grid(True)
    plt.show()
    plt.pause(0.1)


def plot_vs_steps(mean, median, std):
    plt.figure()
    x = np.arange(np.shape(mean)[0])
    plt.plot(x, mean, 'r-', lw=2, label='mean')
    plt.plot(x, median, 'b-', lw=2, label='median')
    plt.plot(x, std, 'k-', lw=2, label='std')
    plt.xlabel('steps')
    plt.ylabel('value')
    plt.title('Cost vs. individual')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.pause(0.1)


def evaluate_knapsack(individuals, items, max_weight):
    fitness_arr = fitness_population(individuals, items, max_weight)
    plot_results(fitness_arr)
    return


num_individuals = 1000
num_items = 200
item_weight_min = 1
item_weight_max = 15
item_cost_min = 1
item_cost_max = 10

# Randomly start individuals
individuals = np.zeros((num_individuals, num_items))
for i in range(num_individuals):
    for j in range(num_items):
        individuals[i, j] = np.random.randint(0, 10*num_items)

# # Doctored testing
# weight_sum = 0
# items = np.zeros((num_items, 2))
# ratio = 0.5
# for i in range(num_items):
#     if i / num_items < ratio:
#         items[i, 0] = item_weight_min
#     else:
#         items[i, 0] = item_weight_max
#     weight_sum += items[i, 0]
#     items[i, 1] = item_cost_max
# max_weight = item_weight_min * num_items * ratio
# print(max_weight)
# print('max cost = ', item_cost_max * num_items * ratio)

# Random testing
weight_sum = 0
items = np.zeros((num_items, 2))
for i in range(num_items):
    items[i, 0] = np.random.randint(item_weight_min, item_weight_max)
    weight_sum += items[i, 0]
    items[i, 1] = np.random.randint(item_cost_min, item_cost_max)
max_weight = 0.5*weight_sum

#evaluate_knapsack(individuals, items, max_weight)
# soln = knapsack_evolution_asexual(individuals, items, max_weight)
soln = knapsack_evolution(individuals, items, max_weight)
#evaluate_knapsack(soln, items, max_weight)
