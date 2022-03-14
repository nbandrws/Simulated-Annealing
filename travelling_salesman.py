# travelling salesman problem
from sa import sa
import numpy as np
import matplotlib.pyplot as plt


# route perturbations
def perturb_rand(r):
    rp = r.copy()

    # pick 2 random cities in route
    c1 = np.random.randint(1, len(rp) - 1)
    c2 = c1
    while c1 is c2:
        c2 = np.random.randint(1, len(rp) - 1)

    # swap their order in the route
    rp[c1], rp[c2] = rp[c2], rp[c1]
    assert rp[0] is rp[-1]
    return rp


# total route distance
def dist(pos, route):
    cost = 0
    for i in range(1, len(route)):
        cost += np.linalg.norm(pos[route[i]] - pos[route[i - 1]])
    return cost


# annealing schedule
def gen_schedule():
    start = 10
    stop = 0
    tstep = 0.1
    numit = 10

    now = start
    schedule = []
    while now >= stop:
        for _ in range(numit):
            schedule.append(now)
        now -= tstep
    return schedule


# for repeatability
np.random.seed(0)

# generate random cities
n = 20  # number of cities
lo_lim = 0
hi_lim = 100
pos = {}
for i in range(n):
    pos[i] = np.random.uniform(lo_lim, hi_lim, 2)

# initial guess
x0 = [i for i in range(n)]
x0.append(0)

# cost function
f = lambda r: dist(pos, r)

# annealing schedule
schedule = gen_schedule()

# simulated annealing test
xstar, fstar, xlog, flog, plog = sa(x0, f, perturb_rand, schedule)
idx = list(range(len(flog)))

# plot cost
plt.figure()
plt.plot(idx, flog)
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('cost vs iteration')

# plot probability
plt.figure()
plt.plot(idx, plog)
plt.xlabel('iteration')
plt.ylabel('probability')
plt.title('annealing schedule')

# plot solution
flag = True
plt.figure()
for stop in pos:
    if flag:
        plt.plot(pos[stop][0], pos[stop][1], 'rs')
        flag = False
    else:
        plt.plot(pos[stop][0], pos[stop][1], 'bs')
for i in range(1, len(xstar)):
    xpos = [pos[xstar[i - 1]][0], pos[xstar[i]][0]]
    ypos = [pos[xstar[i - 1]][1], pos[xstar[i]][1]]
    plt.plot(xpos, ypos, 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('optimal route')

# show figures
plt.show()
