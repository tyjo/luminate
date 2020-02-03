import numpy as np
from scipy.integrate import RK45, solve_ivp
from scipy.special import logsumexp

pseudo_count = 1e-5

def elastic_net_lotka_volterra(X, U, T, Q_inv, r_A, r_g, r_B, alpha=1, tol=1e-3, verbose=False, max_iter=100000):

    def gradient(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        grad = Q_inv.dot(f.T.dot(pgu_stacked))

        # l2 regularization terms
        A = AgB[:,:xDim]
        g = AgB[:,xDim:(xDim+1)]
        B = AgB[:,(xDim+1):]
        grad[:,:xDim] += -2*(1-r_A)*alpha*A
        grad[:,xDim:(xDim+1)] += -2*(1-r_g)*alpha*g
        grad[:,(xDim+1):] += -2*(1-r_B)*alpha*B
        return -grad


    def generalized_gradient(AgB, grad, step):
            proximal = prv_AgB - step*grad

            # threshold A
            A = proximal[:,:xDim]
            A[A < -step*alpha*r_A] += step*alpha*r_A
            A[A > step*alpha*r_A] -= step*alpha*r_A
            A[np.logical_and(A >= -step*alpha*r_A, A <= step*alpha*r_A)] = 0

            # threshold g
            g = proximal[:,xDim:(xDim+1)]
            g[g < -step*alpha*r_g] += step*alpha*r_g
            g[g > step*alpha*r_g] -= step*alpha*r_g
            g[np.logical_and(g >= -step*alpha*r_g, g <= step*alpha*r_g)] = 0

            # threshold B
            B = proximal[:,(xDim+1):]
            B[B < -step*alpha*r_B] += step*alpha*r_B
            B[B > step*alpha*r_B] -= step*alpha*r_B
            B[np.logical_and(B >= -step*alpha*r_B, B <= step*alpha*r_B)] = 0

            AgB_proximal = np.zeros(AgB.shape)
            AgB_proximal[:,:xDim] = A
            AgB_proximal[:,xDim:(xDim+1)] = g
            AgB_proximal[:,(xDim+1):] = B

            return (AgB - AgB_proximal)/step


    def objective(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        obj = -0.5*(f.dot(Q_inv)*f).sum()

        return -obj


    def stack_observations(X, U, T):
        # number of observations by xDim
        x_stacked = None
        # number of observations by xDim + 1 + uDim
        pgu_stacked = None
        for x, u, times in zip(X, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                pt0 = np.exp(x[t-1])
                gt0 = np.ones(1)
                ut0 = u[t-1]

                pgu = np.concatenate((pt0, gt0, ut0))

                if x_stacked is None:
                    x_stacked = x[t] - x[t-1]
                    pgu_stacked = dt*pgu

                else:
                    x_stacked = np.vstack((x_stacked, x[t] - x[t-1]))
                    pgu_stacked = np.vstack((pgu_stacked, dt*pgu))
        return x_stacked, pgu_stacked


    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]
    AgB = np.zeros(( xDim, xDim + 1 + uDim ))
    x_stacked, pgu_stacked = stack_observations(X, U, T)
    prv_obj = np.inf
    obj = objective(AgB, x_stacked, pgu_stacked)

    it = 0
    while np.abs(obj - prv_obj) > tol:
        np.set_printoptions(suppress=True)
        grad = gradient(AgB, x_stacked, pgu_stacked)
        prv_AgB = np.copy(AgB)
        prv_obj = obj
        obj = objective(AgB, x_stacked, pgu_stacked)

        step = 0.001
        gen_grad = generalized_gradient(AgB, grad, step)
        while obj > prv_obj - step*(grad*gen_grad).sum() + step/2*np.square(gen_grad).sum() or obj > prv_obj:
            update = prv_AgB - step*grad
            step /= 2
            gen_grad = generalized_gradient(AgB, grad, step)

            # threshold A
            A = update[:,:xDim]
            A[A < -step*alpha*r_A] += step*alpha*r_A
            A[A > step*alpha*r_A] -= step*alpha*r_A
            A[np.logical_and(A >= -step*alpha*r_A, A <= step*alpha*r_A)] = 0

            # threshold g
            g = update[:,xDim:(xDim+1)]
            g[g < -step*alpha*r_g] += step*alpha*r_g
            g[g > step*alpha*r_g] -= step*alpha*r_g
            g[np.logical_and(g >= -step*alpha*r_g, g <= step*alpha*r_g)] = 0

            # threshold B
            B = update[:,(xDim+1):]
            B[B < -step*alpha*r_B] += step*alpha*r_B
            B[B > step*alpha*r_B] -= step*alpha*r_B
            B[np.logical_and(B >= -step*alpha*r_B, B <= step*alpha*r_B)] = 0

            AgB[:,:xDim] = A
            AgB[:,xDim:(xDim+1)] = g
            AgB[:,(xDim+1):] = B

            obj = objective(AgB, x_stacked, pgu_stacked)

        if obj > prv_obj:
            print("Warning: increasing objective", file=sys.stderr)
            print("\tWas:", prv_obj, "Is:", obj, file=sys.stderr)
            break

        obj = objective(AgB, x_stacked, pgu_stacked)
        it += 1

        if verbose:
            print("\t", it, obj)

        if it > max_iter:
            print("Warning: maximum number of iterations ({}) reached".format(max_iter), file=sys.stderr)
            break

    A = AgB[:,:xDim]
    g = AgB[:,xDim:(xDim+1)].flatten()
    B = AgB[:,(xDim+1):]

    return A, g, B


def estimate_elastic_net_regularizers_cv_lotka_volterra(Y, U, T, folds=5):
    ratio = [0, 0.1, 0.5, 0.7, 0.9]
    alphas = [0.1, 1, 10]
    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in ratio:
            for r_g in ratio:
                for r_B in ratio:
                    alpha_rA_rg_rB.append( (alpha, r_A, r_g, r_B ) )
    
    best_r = 0
    best_sqr_err = np.inf
    for alpha, r_A, r_g, r_B in alpha_rA_rg_rB:
            sqr_err = 0
            for fold in range(folds):
                train_Y = []
                train_U = []
                train_T = []
                test_Y = []
                test_U = []
                test_T = []
                for i in range(len(Y)):
                    if i % folds == fold:
                        test_Y.append(Y[i])
                        test_U.append(U[i])
                        test_T.append(T[i])
                    else:
                        train_Y.append(Y[i])
                        train_U.append(U[i])
                        train_T.append(T[i])
                train_X = estimate_log_space_from_observations(train_Y)
                Q_inv = np.eye(train_X[0].shape[1])
                A, g, B = elastic_net_lotka_volterra(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha=alpha, tol=1e-3)
                sqr_err += compute_prediction_error_glv(test_Y, test_U, test_T, A, g, B)

                if np.isnan(sqr_err):
                    continue
                    
            if sqr_err < best_sqr_err:
                best_r = (alpha, r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def estimate_glv_parameters(Y, U, T):
    alpha, r_A, r_g, r_B = estimate_elastic_net_regularizers_cv_lotka_volterra(Y, U, T, folds=len(Y))
    X = estimate_log_space_from_observations(Y)
    Q_inv = np.eye(X[0].shape[1])
    A, g, B = elastic_net_lotka_volterra(X, U, T, Q_inv, r_A, r_g, r_B, alpha)
    return A, g, B


def predict_glv(y0, u, times, A, g, B):
    y0 = np.copy(y0)
    mass = y0.sum()
    y0 /= y0.sum()
    y0 = (y0 + pseudo_count) / (y0 + pseudo_count).sum()
    y0 = mass*y0
    mu = np.log(y0)
    zt = mu
    y_pred = np.zeros((times.shape[0], y0.size))
    y_idx = 0
    for i in range(1,times.shape[0]):
        dt = times[i] - times[i-1]
        zt = zt + dt*(g + A.dot(np.exp(zt)) + B.dot(u[i-1]))
        pt  = np.exp(zt - logsumexp(zt))
        y_pred[i] = pt  
        y_idx += 1
    return y_pred


def compute_square_error(true, est):
    true = np.copy(true)
    est = np.copy(est)
    true /= true.sum()
    est /= est.sum()
    true = (true + pseudo_count) / (true + pseudo_count).sum()
    est = (est + pseudo_count) / (est + pseudo_count).sum()

    return np.square(true - est).sum()


def compute_prediction_error_glv(Y, U, T, A, g, B):
    def compute_total_square_error(y, y_pred):
        err = 0
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err += compute_square_error(yt, ypt)
        return err
    err = 0
    for y, u, t in zip(Y, U, T):
        y_pred = predict_glv(y[0], u, t, A, g, B)
        err += compute_total_square_error(y, y_pred)
    return np.array(err)


def estimate_log_space_from_observations(Y):
    X = []
    for y in Y:
        x = np.zeros((y.shape[0], y.shape[1]))
        for t in range(y.shape[0]):
            mass = y[t].sum()
            pt = y[t] / y[t].sum()
            pt = (pt + pseudo_count) / (pt + pseudo_count).sum()
            yt = mass*pt
            x[t] = np.log(yt)
            assert np.all(np.isfinite(x[t]))
        X.append(x)
    return X


def grad_fn(A, g, B, u):
    def fn(t, x):
        return g + A.dot(np.exp(x)) + B.dot(u)
    return fn


def simulate_glv_trajectory(y0, A, g, B, u, t):
    """Simulates data under generalized Lotka-Volterra.
    """
    y0 = np.copy(y0)
    mass = y0.sum()
    y0 /= y0.sum()
    y0 = (y0 + pseudo_count) / (y0 + pseudo_count).sum()
    y0 = mass*y0
    mu = np.log(y0)
    ntaxa = A.shape[0]
    N  = 10000 # sequencing reads parameter

    x = []
    Xp = []
    xt = mu
    for i in range(t.shape[0]):
        x.append(xt)
        Xp.append(np.exp(xt - logsumexp(xt)))

        dt = 1
        grad = grad_fn(A, g, B, u[i])
        ivp = solve_ivp(grad, (0,0+dt), xt, method="RK45")
        xt = ivp.y[:,-1]

    t = np.array([i for i in range(t.shape[0])])
    x = np.array(x)
    Xp = np.array(Xp)
    return x, Xp, t


def compute_cov(X):
    cov = np.zeros((X[0].shape[1]-1, X[0].shape[1]-1))
    for x in X:
        for t in range(1, x.shape[0]):
            alr_t0 = x[t-1,:-1] - x[t-1,-1]
            alr_t = x[t,:-1] - x[t,-1]
            cov += np.eye(x.shape[1]-1).dot(np.diag(np.outer(alr_t - alr_t0, alr_t - alr_t0))) / (x.shape[0] - 1)
    cov /= len(X)
    return cov


def simulate_counts(X, signal_noise_ratio):
    dim = X[0].shape[1]-1
    cov = compute_cov(X)
    eta = np.diag(cov) / signal_noise_ratio
    Z = []
    Y = []

    for x in X:
        z = []
        y = []
        for t in range(x.shape[0]):
            xt = x[t]
            zt = np.random.normal(loc=xt[:-1]-xt[-1], scale=np.sqrt(eta))
            zt = np.concatenate((zt, np.zeros(1)))
            pt = np.exp(zt - logsumexp(zt))

            # simulate total number of reads with over-dispersion
            N = 10000
            logN = np.random.normal(loc=np.log(N), scale=0.5)
            Nt = np.random.poisson(np.exp(logN))
            yt = np.random.multinomial(Nt, pt).astype(float)

            z.append(zt)
            y.append(yt)
        Z.append(np.array(z))
        Y.append(np.array(y))

    return Z, Y


def simulate(Y, U, T, A, g, B, signal_noise_ratio):
    ntaxa = Y[0][0].size
    X_sim = []
    P_sim = []
    U_sim = []
    T_sim = []

    for n in range(len(Y)):
        x, p, t = simulate_glv_trajectory(Y[n][0], A, g, B, U[n], T[n])
        X_sim.append(x)
        P_sim.append(p)
        U_sim.append(np.copy(U[n]))
        T_sim.append(t)

    Z_sim, Y_sim = simulate_counts(X_sim, signal_noise_ratio)
    return P_sim, Y_sim, U_sim, T_sim


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



def write_effects(U, T, outfile):
    if outfile[-4:] != ".csv":
        outfile += ".csv"

    f = open(outfile, "w")
    f.write("ID,eventID,startDay,endDay\n")
    for i, u in enumerate(U):
        in_effect = False
        t = T[i]
        effect = u.flatten() == 1
        indices = np.argwhere(effect)
        if indices.size > 0:
            start_idx = indices[0]
            end_idx = indices[-1]
            start_day = t[start_idx[0]]
            end_day = t[end_idx[0]]
            f.write("{},{},{},{}\n".format(float(i+1), "event", start_day, end_day))


def sparsify(P, Y, U, T, s):
    assert s > 1
    P_sparse = []
    Y_sparse = []
    U_sparse = []
    T_sparse = []
    for i,y in enumerate(Y):
        p = P[i]
        u = U[i]
        t = T[i]
        tpts = [0]
        while tpts[-1] < y.shape[0]:
            time_to_nxt_obs = np.clip(np.random.poisson(s), s-1, s+1)
            nxt_tpt = tpts[-1] + time_to_nxt_obs
            tpts.append(nxt_tpt)
        tpts = tpts[:-1] # last tpts occurs after end of sequence

        P_sparse.append(p[tpts])
        Y_sparse.append(y[tpts])
        U_sparse.append(u[tpts])
        T_sparse.append(t[tpts])

    return P_sparse, Y_sparse, U_sparse, T_sparse


if __name__ == "__main__":
    np.random.seed(123538)
    import sys
    sys.path.append("../src")
    import util

    IDS, Y_cdiff, U_cdiff, T_cdiff, event_names = util.load_observations(
                                        "../datasets/bucci2016mdsine/cdiff.csv",
                                        "../datasets/bucci2016mdsine/cdiff-events.csv"
                                    )
    IDS, Y_diet, U_diet, T_diet, event_names = util.load_observations(
                                    "../datasets/bucci2016mdsine/diet.csv",
                                    "../datasets/bucci2016mdsine/diet-events.csv"
                                )
    # learn model parameters on two datasets
    try:
        A_cdiff = np.loadtxt("A-cdiff")
        g_cdiff = np.loadtxt("g-cdiff")
        B_cdiff = np.loadtxt("B-cdiff")
        B_cdiff = np.zeros(B_cdiff.shape) # these do not impact simulations
        B_cdiff = np.expand_dims(B_cdiff, axis=1)
    except OSError:
        A_cdiff, g_cdiff, B_cdiff = estimate_glv_parameters(Y_cdiff, U_cdiff, T_cdiff)
        np.savetxt("A-cdiff", A_cdiff)
        np.savetxt("g-cdiff", g_cdiff)
        np.savetxt("B-cdiff", B_cdiff)

    try:
        A_diet = np.loadtxt("A-diet")
        g_diet = np.loadtxt("g-diet")
        B_diet = np.loadtxt("B-diet")
        B_diet = np.expand_dims(B_diet, axis=1)
    except OSError:
        A_diet, g_diet, B_diet = estimate_glv_parameters(Y_diet, U_diet, T_diet)
        np.savetxt("A-diet", A_diet)
        np.savetxt("g-diet", g_diet)
        np.savetxt("B-diet", B_diet)


    signal_noise_ratios = [0.5, 1, 2, 4]
    sparsity = [2, 3, 4, 5, 6]
    for snr in signal_noise_ratios:
        P, Y, U, T = simulate(Y_cdiff, U_cdiff, T_cdiff, A_cdiff, g_cdiff, B_cdiff, snr)
        write_table(Y, T, "datasets/cdiff-dense-snr{}-counts".format(snr))
        write_table(P, T, "datasets/cdiff-dense-snr{}-truth".format(snr))

        for s in sparsity:
            P_sparse, Y_sparse, U_sparse, T_sparse = sparsify(P,Y,U,T,s)
            write_table(Y_sparse, T_sparse, "datasets/cdiff-sparse{}-snr{}-counts".format(s, snr))
            write_table(P_sparse, T_sparse, "datasets/cdiff-sparse{}-snr{}-truth".format(s, snr))


        P, Y, U, T = simulate(Y_diet, U_diet, T_diet, A_diet, g_diet, B_diet, snr)
        write_table(Y, T, "datasets/diet-dense-snr{}-counts".format(snr))
        write_table(P, T, "datasets/diet-dense-snr{}-truth".format(snr))
        write_effects(U,T,"datasets/diet-effects-snr{}".format(snr))

        for s in sparsity:
            P_sparse, Y_sparse, U_sparse, T_sparse = sparsify(P,Y,U,T,s)
            write_table(Y_sparse, T_sparse, "datasets/diet-sparse{}-snr{}-counts".format(s, snr))
            write_table(P_sparse, T_sparse, "datasets/diet-sparse{}-snr{}-truth".format(s, snr))
            write_effects(U,T,"datasets/diet-effects-sparse{}-snr{}".format(s, snr))

