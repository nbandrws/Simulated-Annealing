# travelling salesman problem
from sa import sa
import numpy as np
import matplotlib.pyplot as plt
import time


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
        stop = 0.01
        w = 100
        for t in np.linspace(start, stop, 200):
            for _ in range(w):
                schedule.append(t)
    elif name == 'log':
        w = 100
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
np.random.seed(1)
savefig = True

# generate random cities
x_pos = [-50, 0, 50]
y_pos = [-50, 0, 50]
n = 3  # number of cities per cluster
lo_lim = -10
hi_lim = 10
pos = {}
idx = 0
for x in x_pos:
    for y in y_pos:
        for i in range(n):
            pos[idx] = np.random.uniform(lo_lim, hi_lim, 2) + np.array([x, y])
            idx += 1

# initial route guess
x0 = [i for i in range(idx)]
x0.append(0)

# cost function
f = lambda r: dist(pos, r)

# annealing schedule
schedule = gen_schedule('log')

# simulated annealing
start_time = time.time()
xstar, fstar, xlog, flog, plog = sa(x0, f, perturb_rand, schedule)
print("--- %s seconds ---" % (time.time() - start_time))
print('optimal distance = {:.2f}'.format(fstar))
idx = list(range(len(flog)))

if False:
    nruns = 10
    fstar_log1 = []
    fstar_log2 = []
    fstar_log3 = []
    fstar_log4 = []
    fstar_log5 = []
    fstar_log6 = []
    for i in range(nruns):
        xstar, fstar, xlog, flog, plog = sa(x0, f, perturb_neighbor, gen_schedule('linear'))
        fstar_log1.append(fstar)
        print('a', i)
    for i in range(nruns):
        xstar, fstar, xlog, flog, plog = sa(x0, f, perturb_neighbor, gen_schedule('log'))
        fstar_log2.append(fstar)
        print('b', i)
    for i in range(nruns):
        xstar, fstar, xlog, flog, plog = sa(x0, f, perturb_neighbor, gen_schedule('reanneal'))
        fstar_log3.append(fstar)
        print('c', i)
    for i in range(nruns):
        xstar, fstar, xlog, flog, plog = sa(x0, f, perturb_rand, gen_schedule('linear'))
        fstar_log4.append(fstar)
        print('d', i)
    for i in range(nruns):
        xstar, fstar, xlog, flog, plog = sa(x0, f, perturb_rand, gen_schedule('log'))
        fstar_log5.append(fstar)
        print('e', i)
    for i in range(nruns):
        xstar, fstar, xlog, flog, plog = sa(x0, f, perturb_rand, gen_schedule('reanneal'))
        fstar_log6.append(fstar)
        print('f', i)
    plt.figure()
    plt.plot(list(range(nruns)), fstar_log1, label='Linear-Flip')
    plt.plot(list(range(nruns)), fstar_log2, label='Log-Flip')
    plt.plot(list(range(nruns)), fstar_log3, label='Reanneal-Flip')
    plt.plot(list(range(nruns)), fstar_log4, label='Linear-Swap')
    plt.plot(list(range(nruns)), fstar_log5, label='Log-Swap')
    plt.plot(list(range(nruns)), fstar_log6, label='Reanneal-Swap')
    plt.xlabel('Run Number')
    plt.ylabel('Optimal Cost')
    plt.title('Optimal Cost vs Number of Runs')
    plt.legend()
    if savefig:
        plt.savefig('cost_vs_runs.pdf')

# plot cost
plt.figure()
plt.plot(idx, flog)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration')
if savefig:
    plt.savefig('cost_vs_iteration.pdf')

# plot probability
plt.figure()
plt.plot(idx, plog)
plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.title('Annealing Schedule Probability')
if savefig:
    plt.savefig('prob.pdf')

# plot probability
plt.figure()
plt.plot(list(range(len(schedule))), schedule)
plt.xlabel('Iteration')
plt.ylabel('Temperature')
plt.title('Annealing Schedule')
if savefig:
    plt.savefig('temp.pdf')

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
    xpos = [pos[xlog[15000][i - 1]][0], pos[xlog[15000][i]][0]]
    ypos = [pos[xlog[15000][i - 1]][1], pos[xlog[15000][i]][1]]
    plt.plot(xpos, ypos, 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimal Route')
if savefig:
    plt.savefig('route.pdf')


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
plt.title('Optimal Route')

# show figures
plt.show()
