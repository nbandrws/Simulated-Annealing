import numpy as np


# simulated annealing - discrete
def sa(x0, f, perturb, schedule):
    # hyperparameters
    kb = 1  # Boltzman constant
    dE = 1  # nominal step size

    # logs
    xlog = [x0]
    flog = [f(x0)]
    plog = [np.exp(-dE / (kb * schedule[0]))]

    x = x0.copy()
    xstar = x.copy()  # best value yet
    for T in schedule:
        # perturb in random direction
        xnew = perturb(x)

        prob = np.exp(-dE / (kb * T))  # upward probability acceptance
        if f(xnew) < f(x):
            # accept draw based on merit (downhill)
            x = xnew.copy()

            # check if new value is best yet
            if f(x) < f(xstar):
                xstar = x.copy()

        elif np.random.uniform(0, 1) <= prob:
            # accept draw based on luck (uphill)
            x = xnew.copy()

        # log things
        xlog.append(x)
        flog.append(f(x))
        plog.append(prob)

    return xstar, f(xstar), xlog, flog, plog
