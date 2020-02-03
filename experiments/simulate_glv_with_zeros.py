import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.special import logsumexp
from scipy.integrate import RK45, solve_ivp


def grad_fn(A, g):
    def fn(t, x):
        return x*(g + A.dot(x))
    return fn


def simulate_glv(A, g, eta, ntaxa, ndays):
    """Simulates data under generalized Lotka-Volterra.

    Parameters
    ----------
        ntaxa  : number of species to simulate
        ndays  : number of days to simulate

    Returns
    -------
        x  : an ndays by ntaxa matrix of latent log abundance
        p  : an ndays by ntaxa matrix of latent relative abundances
        y  : an ndays by ntaxa matrix of observed sequencing counts

    """
    N  = 10000 # sequencing reads parameter

    x = []
    z = []
    Xp = []
    y = []
    mu = np.ones(ntaxa)
    mu[1] = 0.25
    mu[2] = 0.4
    mu = np.abs(np.random.multivariate_normal(mean=mu, cov=np.eye(ntaxa)))
    for t in range(ndays):
        xt = mu
        xtp = xt/xt.sum()
        zt = np.random.normal(loc=np.log(xt[:-1]/xt[-1]), scale=np.sqrt(eta))
        zt = np.concatenate((zt, np.zeros(1)))
        pt = np.exp(zt - logsumexp(zt))
        pt /= pt.sum()

        # zero out taxa 1 for extinction
        if xtp[0] < 1e-4:
            pt[0] = 0
            xtp[0] = 0
        pt[1] = np.max((1./5000, pt[1]))
        xtp[1] = np.max((1./5000, xtp[1]))

        # simulate total number of reads with over-dispersion
        logN = np.random.normal(loc=np.log(N), scale=0.5)
        Nt = np.random.poisson(np.exp(logN))
        yt = np.random.multinomial(5000, pt).astype(float)



        x.append(xt)
        z.append(zt)
        Xp.append(xtp)
        y.append(yt)

        grad = grad_fn(A, g)
        ivp = solve_ivp(grad, (0,1), xt, method="RK45")
        mu = ivp.y[:,-1]

    x = np.array(x)
    z = np.array(z)
    Xp = np.array(Xp)
    y = np.array(y)
    return x, z, y, Xp


def compute_cov(X):
    cov = np.zeros((X[0].shape[1], X[0].shape[1]))
    for x in X:
        for t in range(1, x.shape[0]):
            cov += np.outer(x[t] - x[t-1], x[t] - x[t-1]) / (x.shape[0] - 1)
    cov /= len(X)
    return cov


def simulate(nsimulations, ntaxa, ndays, days_between):
    A = np.random.normal(loc=0, scale=0.025, size=(ntaxa,ntaxa))
    for i,row in enumerate(A):
        for j,a in enumerate(row):
            if i == j:
                A[i,j] = -np.abs(a)
            else:
                A[i,j] = -np.abs(a)
            if np.random.uniform() < 0.5:
                A[i,j] = 0
    g  = np.abs(np.random.normal(loc=0,scale=0.025,size=(ntaxa)))
    eta = np.random.uniform(low=0.01, high=0.05, size=ntaxa-1)
    X = []
    P = []
    Y = []
    T = []
    for n in range(nsimulations):
        t_pts = np.array([i for i in range(ndays)])
        x, z, y, p = simulate_glv(A, g, eta, ntaxa, ndays)
        while np.any(np.isnan(x)):
            x, z, y, p = simulate_glv(A, g, eta, ntaxa, ndays)
        X.append(x)
        P.append(p)
        Y.append(y)
        T.append(t_pts)
    cov = compute_cov(X)
    return P, Y, T


def simulate_zeros(nsimulations, ndays, days_between):
    # simulate 4 taxa
    ntaxa = 4
    A = np.array([
        [   0,   0,     0,    0],
        [   0,  -1,    -1,    0],
        [   0,   0 , -0.5, -0.4],
        [   0,   0 , -0.4, -0.5]
    ])
    g = np.array([-0.5, 0.2, 0.5, 0.5])
    eta = np.random.uniform(low=0.01, high=0.05, size=ntaxa-1)
    X = []
    P = []
    Y = []
    T = []
    for n in range(nsimulations):
        t_pts = np.array([i for i in range(ndays)])
        x, z, y, p = simulate_glv(A, g, eta, ntaxa, ndays)
        X.append(x)
        P.append(p)
        Y.append(y)
        T.append(t_pts)
    cov = compute_cov(X)
    return P, Y, T


def plot_bar(ax, y):
    T = y.shape[0]
    cm = plt.get_cmap("tab20c")
    colors = [cm(i) for i in range(20)]
    time = np.array([t for t in range(T)])
    width = 1
    ax.bar(time, y[:,0], width=width, color=colors[0])
    for j in range(1, y.shape[1]):
        ax.bar(time, y[:,j], bottom=y[:,:j].sum(axis=1), width=width, color=colors[j % 20])


def plot_trajectories(Y, outfile):
    N = len(Y)
    fig, ax = plt.subplots(nrows=N, ncols=1, figsize=(10,20))
    for i in range(N):
        plot_bar(ax[i], (Y[i].T / Y[i].sum(axis=1)).T)

    if outfile[-4:] == ".pdf":
        plt.savefig(outfile)
    else:
        plt.savefig(outfile + ".pdf")
    plt.close()


def write_table(observations, time_points, outfile):
    ntaxa = observations[0].shape[1]
    otu_table = []
    for idx, obs in enumerate(observations):
        for t,obs_t in enumerate(obs):
            header = np.array([idx+1, time_points[idx][t]])
            row = np.concatenate((header, obs_t))
            otu_table.append(row)
    otu_table = np.array(otu_table).T
    col1 = np.array(["id", "time"] + ["OTU " + str(i) for i in range(ntaxa)])
    col1 = np.expand_dims(col1, axis=1)
    otu_table = np.hstack((col1, otu_table.astype(str)))

    if outfile[-4:] == ".csv":
        np.savetxt(outfile, otu_table, delimiter=",", fmt="%s")
    else:
        np.savetxt(outfile + ".csv", otu_table, delimiter=",", fmt="%s")


def compute_percent_zeros(P, Y):
    true_zeros = 0
    sampling_zeros = 0
    for p,y in zip(P, Y):
        true_zeros += p[p == 0].size

        sampling_zeros += np.logical_and(p != 0, y == 0).sum()
    return true_zeros, sampling_zeros


if __name__ == "__main__":
    np.random.seed(28091)
    ntaxa = [4]
    ndays = [30]
    for d in ntaxa:
        for n in ndays:
            P, Y, T = simulate_zeros(100, n, 1)
            write_table(Y, T, "datasets/ntaxa{}-ndays{}-dense-zeros-counts".format(d, n))
            write_table(P, T, "datasets/ntaxa{}-ndays{}-dense-zeros-truth".format(d, n))
            tr_zero, s_zero = compute_percent_zeros(P, Y)
            print("True Zeros:", tr_zero)
            print("Sampling Zeros:", s_zero)