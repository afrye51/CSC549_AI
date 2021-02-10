import numpy as np
import matplotlib.pyplot as plt


def perturb_list(cities_in, window, method='randomize'):
    cities = np.copy(cities_in)
    n = np.shape(cities)[0]
    if method == 'randomize':
        lst = np.arange(num)
        np.random.shuffle(lst)
        cities = cities[lst]

    elif method == 'swap':
        a = np.random.randint(0, n)
        b = np.random.randint(0, n)
        while b == a:
            b = np.random.randint(0, n)
        temp = cities[a]
        cities[a] = cities[b]
        cities[b] = temp

    elif method == 'swap sublists':
        sub_len = np.random.randint(0, n // 2)
        a = np.random.randint(0, n - sub_len*2)
        b = np.random.randint(a + sub_len, n - sub_len)
        temp = cities[a:a + sub_len]
        cities[a:a + sub_len] = cities[b:b + sub_len]
        cities[b:b + sub_len] = temp

    elif method == 'invert sublist':
        # n = 2 to n, large T means n small T means 2
        sub_len = np.random.randint(2, window)
        a = np.random.randint(0, n - sub_len)
        cities[a:a + sub_len] = cities[a:a + sub_len][::-1]

    elif method == 'randomize sublist':
        sub_len = np.random.randint(0, n)
        a = np.random.randint(0, n - sub_len)
        temp = cities[a:a + sub_len]
        temp = perturb_list(temp, 'randomize')
        cities[a:a + sub_len] = temp

    else:
        print('invalid method, nothing done. ')

    return cities


def city_dist(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)


def eval_distance(cities, locs):
    loop_cities = np.append(cities, cities[0])
    n = np.shape(loop_cities)[0]
    dist = 0
    for i in range(n-1):
        this_dist = city_dist(locs[loop_cities[i]], locs[loop_cities[i+1]])
        dist += this_dist
    return dist


def traveling_salesman(init_cities, locs):
    cities = np.copy(init_cities)
    k = 0
    T = 3000
    T_min = 0.0001
    T_mult = 0.999
    k_max = int(np.ceil(-np.log(T / T_min) / np.log(T_mult))) + 1
    dist = eval_distance(cities, locs)
    dist_arr = np.zeros(k_max)
    dist_arr[k] = dist
    k_tenth = k_max // 10
    n = np.shape(cities)[0]
    T_log = np.log(T / T_min)
    divisor = (n - 3) / T_log
    while k < k_max and T > T_min:
        # if k % k_tenth == 0:
        #     print('k = ', k)
        operating_window = int(np.log(T / T_min) * divisor) + 3
        cities_cand = perturb_list(cities, operating_window, 'invert sublist')
        dist_cand = eval_distance(cities_cand, locs)
        dist = eval_distance(cities, locs)
        try:
            ev = np.exp((dist - dist_cand)/T)
        except:
            print('overflow')
        rnd = np.random.rand()
        if rnd < ev:
            cities = np.copy(cities_cand)
        k += 1
        T *= T_mult
        dist_arr[k] = dist
    return cities, dist_arr


def cost_hash(start, arr):
    return str(start) + ':' + str(arr)


class TSP:
    def __init__(self, cities, locs):
        # Algorithm taken from https://www.baeldung.com/cs/tsp-dynamic-programming

        # Compute distances between each city
        n = np.shape(cities)[0]
        self.dist = np.zeros((n, n))
        self.cost = []
        for i in range(n):
            for j in range(i + 1, n):
                self.dist[i, j] = city_dist(locs[i], locs[j])
                self.dist[j, i] = self.dist[i, j]
        self.Visited = dict()
        self.Cost = dict()

    def compute_tsp(self, s, N):
        if len(N) == 1:
            self.Cost[cost_hash(s, N)] = [self.dist[s, N[0]], np.array([s, N[0]])]
            return
        else:
            min_cost = [np.inf, [0]]
            for i in N:
                index = np.where(N == i)
                new_list = np.delete(N, index)

                if cost_hash(i, new_list) not in self.Cost:
                    self.compute_tsp(i, new_list)

                this_cost = self.Cost[cost_hash(i, new_list)][0] + self.dist[s, i]
                if this_cost < min_cost[0]:
                    min_cost = [this_cost, np.insert(N, 0, s)]

                self.Cost[cost_hash(s, N)] = min_cost
        return

def plot_x_vs_k(x, name):
    plt.figure()
    k = np.arange(np.shape(x)[0])
    plt.plot(k, x, '-', lw=2)
    plt.xlabel('iteration')
    plt.ylabel('x')
    plt.title(name)
    plt.grid(True)
    plt.show()
    plt.pause(0.1)


def plot_results(cities, locs):
    plt.figure()
    loop_cities = np.append(cities, cities[0])
    ordered = locs[loop_cities, :]
    x = ordered[:, 0]
    y = ordered[:, 1]
    plt.plot(x, y, 'k.', lw=5)
    plt.plot(x, y, 'r-', lw=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cities')
    plt.grid(True)
    plt.show()
    plt.pause(0.1)


num = 100
minim = -10
maxim = 10
cities = np.arange(num)
np.random.shuffle(cities)

locs = np.zeros((num, 2))
for i in range(num):
    locs[i] = (np.random.rand(1, 2) * (maxim - minim)) + minim

# num = 5
# locs = np.array([[0, -2], [-2, -1], [-1, 2], [1, 2], [2, -1]])
# # min = 12.79669127533634
# num = 16
# locs = np.array([[2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [-1, 2], [-2, 2], [-2, 1],
#                  [-2, 0], [-2, -1], [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2], [2, -1]])
# min = 16
min_dist = [np.inf, [0]]

for i in range(100):
    print(i)
    cities_solved, dist_results = traveling_salesman(cities, locs)
    this_dist = eval_distance(cities_solved, locs)
    print(this_dist)
    print()
    if this_dist < min_dist[0]:
        min_dist = [this_dist, cities_solved]

plot_results(min_dist[1], locs)
