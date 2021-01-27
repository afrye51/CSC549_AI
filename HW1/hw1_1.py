import numpy as np
import matplotlib.pyplot as plt


def update_x_stochastic(x, mean, std, minim=0, maxim=1):
    step = np.random.normal(mean, std)
    x = x + step
    x = max(min(maxim, x), minim)
    return x


def update_x_step(x, step, minim=0, maxim=1):
    x = x + step
    x = max(min(maxim, x), minim)
    return x


def eval_func(x):
    return (-2 ** (-2 * ((x - 0.1) / 0.9) ** 2)) * (np.sin(5 * np.pi * x) ** 6)


def eval_derivative(x):
    return 2**(-2.46914*(x-0.1)**2) * np.sin(5*np.pi*x)**5 * ((3.42295*x-0.342295)*np.sin(5*np.pi*x) - 94.2478*np.cos(5*np.pi*x))


def perturb(T):
    # dx = (T + 273) / 1000000
    dx = (T + 0.01) ** 0.25 / 600
    dx *= (np.random.randint(2) - 0.5) * 2
    return dx


def simulated_annealing(x_0):
    k = 0
    T = 3000
    T_min = 0.0001
    T_mult = 0.999
    k_max = int(np.ceil(-np.log(T / T_min) / np.log(T_mult))) + 1
    x = x_0
    e_x = eval_func(x)
    k_arr = np.arange(k_max)
    x_arr = np.zeros(k_max)
    x_arr[k] = x
    while k < k_max and T > T_min:
        # xp = update_x_stochastic(x, 0, perturb(T))
        xp = update_x_step(x, perturb(T))
        e_xp = eval_func(xp)
        e_x = eval_func(x)
        try:
            ev = np.exp((e_x - e_xp)/T)
        except:
            print('overflow')
        rnd = np.random.rand()
        if rnd < ev:
            x = xp
        k += 1
        T *= T_mult
        x_arr[k] = x
    return x, x_arr


def newtons_method(x_0, k_max=15000):
    k = 0
    x = x_0
    x_last = x + 1
    x_arr = []
    x_arr.append(x)
    while k < k_max and abs(x_last - x) > 0.0001:
        d_x = eval_derivative(x)
        d_xpdx = eval_derivative(x + 0.000001)
        d2_x = (d_xpdx - d_x) / 0.000001
        x_last = x
        x = x - (d_x / d2_x)
        x_arr.append(x)
    return x, x_arr


def gradient_descent(x_0, x_step=0.0001, k_max=15000):
    k = 0
    x = x_0
    x_arr = []
    x_arr.append(x)
    while k < k_max:
        e_x = eval_func(x)
        x_high = update_x_step(x, x_step)
        x_low = update_x_step(x, -x_step)
        if eval_func(x_high) < e_x:
            x = x_high
        elif eval_func(x_low) < e_x:
            x = x_low
        else:
            break
        x_arr.append(x)
    return x, x_arr


def plot_x_vs_k(x, name):
    k = np.arange(np.shape(x)[0])
    plt.plot(k, x, '-', lw=2)
    plt.xlabel('iteration')
    plt.ylabel('x')
    plt.title(name)
    plt.grid(True)
    plt.show()


def plot_results(x_0, x_f, name):
    plt.plot(x_0, x_f, '.', lw=2)
    plt.xlabel('Initial x value')
    plt.ylabel('Final x value')
    plt.title(name)
    plt.grid(True)
    plt.show()


num = 100
minim = 0
maxim = 1
x_0 = np.linspace(minim, maxim, num)
x_f = np.zeros(num)

# __, x_arr = simulated_annealing(x_0[0])
# plot_x_vs_k(x_arr, 'simulated annealing')
# for i in range(num):
#     x_f[i], __ = simulated_annealing(x_0[i])
# plot_results(x_0, x_f, 'simulated annealing')

__, x_arr = newtons_method(x_0[32])
plot_x_vs_k(x_arr, 'newtons method')
for i in range(num):
    x_f[i], __ = newtons_method(x_0[i])
plot_results(x_0, x_f, 'newtons method')

__, x_arr = gradient_descent(x_0[0])
plot_x_vs_k(x_arr, 'gradient descent')
for i in range(num):
    x_f[i], __ = gradient_descent(x_0[i])
plot_results(x_0, x_f, 'gradient descent')
