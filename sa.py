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

    xstar = x0.copy()
    for T in schedule:
        # perturb in random direction
        xnew = perturb(xstar)

        prob = np.exp(-dE / (kb * T))  # upward probability acceptance
        if f(xnew) < f(xstar):
            # accept draw based on merit
            xstar = xnew
        elif np.random.uniform(0, 1) <= prob:
            # accept draw based on luck
            xstar = xnew

        # log things
        xlog.append(xstar)
        flog.append(f(xstar))
        plog.append(prob)

    return xstar, f(xstar), xlog, flog, plog
