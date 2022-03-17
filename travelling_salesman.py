# travelling salesman problem
from sa import sa
import numpy as np
import matplotlib.pyplot as plt


# perturb: random swap
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


# perturb: swap neighbors
def perturb_neighbor(r):
    rp = r.copy()

    # pick random city in route
    c1 = np.random.randint(1, len(rp) - 1)

    # check if 2nd or 2nd to last; can't swap with end points
    if c1 == 1:
        # 2nd point; can only swap forward
        rp[c1], rp[c1 + 1] = rp[c1 + 1], rp[c1]
    elif c1 == len(rp) - 2:
        # 2nd to last; can only swap backward
        rp[c1], rp[c1 - 1] = rp[c1 - 1], rp[c1]
    else:
        # random draw for swap direction
        flip = np.random.randint(0, 2)
        if flip == 0:
            # swap forward
            rp[c1], rp[c1 + 1] = rp[c1 + 1], rp[c1]
        elif flip == 1:
            # swap backward
            rp[c1], rp[c1 - 1] = rp[c1 - 1], rp[c1]
    return rp


# total route distance
def dist(pos, route):
    cost = 0
    for i in range(1, len(route)):
        cost += np.linalg.norm(pos[route[i]] - pos[route[i - 1]])
    return cost


# annealing schedule
def gen_schedule(name):
    schedule = []
    if name == 'linear':
        start = 10
        stop = 0
        tstep = 0.1
        w = 10

        now = start
        while now >= stop:
            for _ in range(w):
                schedule.append(now)
            now -= tstep
    elif name == 'log':
        w = 200
        for x in np.linspace(0.0001, 7, 200):
            for _ in range(w):
                schedule.append(1 - np.log10(x))
    elif name == 'reanneal':
        w = 100
        for x in np.linspace(0.0001, 7, 200):
            for _ in range(w):
                schedule.append(1 - np.log10(x) + 0.1 * np.sin(10 * x))
    else:
        assert False, 'Invalid schedule name'
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

# initial route guess
x0 = [i for i in range(n)]
x0.append(0)

# perturb neighbor test
# for i in range(10):
#     rp = perturb_neighbor(x0)
#     print(rp)

# cost function
f = lambda r: dist(pos, r)

# annealing schedule
schedule = gen_schedule('reanneal')

# simulated annealing test
xstar, fstar, xlog, flog, plog = sa(x0, f, perturb_rand, schedule)
print('optimal distance = {:.2f}'.format(fstar))
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
plt.title('annealing probability')

# plot probability
plt.figure()
plt.plot(list(range(len(schedule))), schedule)
plt.xlabel('iteration')
plt.ylabel('temperature')
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
