import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# simulated annealing
def anneal(f, low, high):
    # initial guess
    x0 = np.random.uniform(low, high)

    # annealing schedule
    T = 10
    Tf = 0
    tstep = 0.1

    # hyperparameters
    kb = 1  # Boltzman constant
    dE = 1  # nominal step size

    # perturbation
    xlog = [x0]
    flog = [f(x0)]
    plog = [np.exp(-dE / (kb * T))]

    xstar = x0
    numit = 0
    while T >= Tf:
        # perturb in random direction and make sure it is within bounds
        xnew = np.inf
        while xnew > high or xnew < low:
            xnew = xstar + dE * np.random.randn()

        prob = np.exp(-dE / (kb * T))  # upward probability acceptance
        if f(xnew) < f(xstar):
            # accept draw based on merit
            xstar = xnew
        elif np.random.uniform(0, 1) <= prob:
            # accept draw based on luck
            xstar = xnew

        # update temperature (linear annealing schedule)
        T -= tstep
        numit += 1

        # log things
        xlog.append(xstar)
        flog.append(f(xstar))
        plog.append(prob)

    return xstar, f(xstar), xlog, flog, plog


def x2(x):
    return x ** 2


def rastrigin(x):
    A = 10
    return A + x ** 2 - A * np.cos(2 * np.pi * x)


f = rastrigin

# solve
np.random.seed(0)
xstar, fxstar, xlog, flog, plog = anneal(f, -5, 5)
print('optimal x =', xstar)
print('optimal value f(x) =', fxstar)

# plotting
x = np.linspace(-5, 5, 1000)
fig, ax = plt.subplots()
ax.plot(x, [f(x) for x in x])
title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
xdata, ydata = [], []
x_past, = plt.plot([], [], 'g.')
x_now, = plt.plot([], [], 'ro')

def init():
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    return x_past, x_now,

def update(i):
    x = xlog[i]
    xdata.append(x)
    ydata.append(f(x))
    x_past.set_data(xdata, ydata)
    x_now.set_data(x, f(x))
    title.set_text('Uphill probability = {:.2f}'.format(plog[i]))
    return x_past, x_now, title,

ani = FuncAnimation(fig, update, frames=len(xlog), init_func=init, blit=True, repeat=False)
plt.show()


ret = dual_annealing(f, bounds=[(-5, 5)])
ret.x