import numpy as np
import sys
from scipy.misc import logsumexp

class CompositionalLotkaVolterra:
    """Inference for compositional Lotka-Volterra.
    """

    def __init__(self, X=None, Y=None, T=None, U=None):
        """
        Parameters
        ----------
            X : A list of T_x by D-1 dimensional numpy arrays of
                estimated additive log-ratios.
            Y  : A list of T_y by D dimensional numpy arrays of
                 time-series observations. T_y denotes the number
                 of observations for sequence y.
            T : A list of T_x by 1 dimensional numpy arrays giving
                the times of each observation x.
            U : An optional list of T_x by P numpy arrays of external
                perturbations for each x.
        """
        self.X = X
        self.Y = Y
        self.T = T
        if U is None and X is not None:
            self.U = [ np.zeros((x.shape[0], 1)) for x in X ]
        else:
            self.U = U

        # Parameter estimates
        self.A = None
        self.g = None
        self.B = None
        self.Q_inv = np.eye(self.X[0].shape[1]) if X is not None else None

        # Regularization parameters
        self.alpha = None
        self.r_A = None
        self.r_g = None
        self.r_B = None


    def train(self, verbose=False):
        """Estimate regularization parameters and CLV model parameters.
        """
        if verbose:
            print("\tEstimating regularizers...", file=sys.stderr)
        self.alpha, self.r_A, self.r_g, self.r_B = estimate_elastic_net_regularizers_cv(self.X, self.Y, self.U, self.T, verbose=verbose)
        
        if verbose:
            print("\tEstimating model parameters...", file=sys.stderr)
        self.A, self.g, self.B = elastic_net_clv(self.X, self.U, self.T, self.Q_inv, self.alpha, self.r_A, self.r_g, self.r_B)


    def predict(self, y0, times, u = None):
        """Predict relative abundances from initial conditions.

        Parameters
        ----------
            y0     : the initial observation, a D-dim numpy array
            times  : a T_x by 1 numpy array of sample times
            u      : a T_x by P numpy array of external perturbations

        Returns
        -------
            y_pred : a T_x by D numpy array of predicted relative
                     abundances. Since we cannot predict initial
                     conditions, the first entry is set to the array
                     of -1.
        """
        if u is None:
            u = np.zeros((x.shape[0], 1))
        x0 = estimate_initial_conditions(y0)
        y_pred = predict(x0, u, times, self.A, self.g, self.B)
        y_pred[0] = np.zeros(y_pred.shape[1])
        return y_pred


    def get_params(self):
        A = np.copy(self.A)
        g = np.copy(self.g)
        B = np.copy(self.B)
        return A, g, B


def estimate_initial_conditions(y0):
    """Estimate additive log-ratio from sequencing counts.

    Parameters
    ----------
        y0 : a D dimensional numpy array of sequencing counts.

    Returns
    -------
        x0 : a D-1 dimensional array of the estimated alr
    """

    y0 /= y0.sum()
    x0 = np.log( (y0[:-1] + 0.001) / (y0[-1] + 0.001) )
    return x0



def elastic_net_clv(X, U, T, Q_inv, alpha, r_A, r_g, r_B,  tol=1e-3, verbose=False, max_iter=100000):
    """Estimate parameters of compositional Lotka Volterra using elastic net regularization.
    """

    def gradient(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        grad = Q_inv.dot(f.T.dot(pgu_stacked))

        # l2 regularization terms
        A = AgB[:,:yDim]
        g = AgB[:,yDim:(yDim+1)]
        B = AgB[:,(yDim+1):]
        grad[:,:yDim] += -2*alpha*(1-r_A)*A
        grad[:,yDim:(yDim+1)] += -2*alpha*(1-r_g)*g
        grad[:,(yDim+1):] += -2*alpha*(1-r_B)*B
        return -grad


    def generalized_gradient(AgB, grad, step):
            proximal = prv_AgB - step*grad

            # threshold A
            A = proximal[:,:yDim]
            A[A < -step*alpha*r_A] += step*alpha*r_A
            A[A > step*alpha*r_A] -= step*alpha*r_A
            A[np.logical_and(A >= -step*alpha*r_A, A <= step*alpha*r_A)] = 0

            # threshold g
            g = proximal[:,yDim:(yDim+1)]
            g[g < -step*alpha*r_g] += step*alpha*r_g
            g[g > step*alpha*r_g] -= step*alpha*r_g
            g[np.logical_and(g >= -step*alpha*r_g, g <= step*alpha*r_g)] = 0

            # threshold B
            B = proximal[:,(yDim+1):]
            B[B < -step*alpha*r_B] += step*alpha*r_B
            B[B > step*alpha*r_B] -= step*alpha*r_B
            B[np.logical_and(B >= -step*alpha*r_B, B <= step*alpha*r_B)] = 0

            AgB_proximal = np.zeros(AgB.shape)
            AgB_proximal[:,:yDim] = A
            AgB_proximal[:,yDim:(yDim+1)] = g
            AgB_proximal[:,(yDim+1):] = B

            return (AgB - AgB_proximal)/step


    def objective(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        obj = -0.5*(f.dot(Q_inv)*f).sum()

        return -obj

    def stack_observations(X, U, T):
        # number of observations by xDim
        x_stacked = None
        # number of observations by yDim + 1 + uDim
        pgu_stacked = None
        for x, u, times in zip(X, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                zt0 = np.concatenate((x[t-1], np.array([0])))
                pt0 = np.exp(zt0 - logsumexp(zt0))
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


    # assert 0 < r_A and r_A < 1
    # assert 0 < r_g and r_g < 1
    # assert 0 < r_B and r_B < 1
    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]
    AgB = np.zeros(( xDim, yDim + 1 + uDim ))

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
            A = update[:,:yDim]
            A[A < -step*alpha*r_A] += step*alpha*r_A
            A[A > step*alpha*r_A] -= step*alpha*r_A
            A[np.logical_and(A >= -step*alpha*r_A, A <= step*alpha*r_A)] = 0

            # threshold g
            g = update[:,yDim:(yDim+1)]
            g[g < -step*alpha*r_g] += step*alpha*r_g
            g[g > step*alpha*r_g] -= step*alpha*r_g
            g[np.logical_and(g >= -step*alpha*r_g, g <= step*alpha*r_g)] = 0

            # threshold B
            B = update[:,(yDim+1):]
            B[B < -step*alpha*r_B] += step*alpha*r_B
            B[B > step*alpha*r_B] -= step*alpha*r_B
            B[np.logical_and(B >= -step*alpha*r_B, B <= step*alpha*r_B)] = 0

            AgB[:,:yDim] = A
            AgB[:,yDim:(yDim+1)] = g
            AgB[:,(yDim+1):] = B

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

    #print("\t", it, obj)
    A = AgB[:,:yDim]
    g = AgB[:,yDim:(yDim+1)].flatten()
    B = AgB[:,(yDim+1):]

    return A, g, B


def estimate_elastic_net_regularizers_cv(X, Y, U, T, folds=10, verbose=False):
    if len(X) == 1:
        print("Error: cannot estimate regularization parameters from single sample", file=sys.stderr)
        exit(1)
    elif len(X) < 10:
        folds = len(X)
    
    #ratio = [0, 0.1, 0.5, 0.7, 0.9]
    rs = [0.1, 0.5, 0.9]
    alphas = [0.1, 1, 10]
    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in rs:
            for r_g in rs:
                for r_B in rs:
                    alpha_rA_rg_rB.append( (alpha, r_A, r_g, r_B ) )
    
    np.set_printoptions(suppress=True)
    best_r = 0
    best_sqr_err = np.inf
    for i, aAgB in enumerate(alpha_rA_rg_rB):
        alpha, r_A, r_g, r_B = aAgB
        print("\tTesting regularization parameter set", i+1, "of", len(alpha_rA_rg_rB), file=sys.stderr)
        sqr_err = 0
        for fold in range(folds):
            train_X = []
            train_Y = []
            train_U = []
            train_T = []

            test_X = []
            test_Y = []
            test_U = []
            test_T = []
            for i in range(len(X)):
                if i % folds == fold:
                    test_X.append(X[i])
                    test_Y.append(Y[i])
                    test_U.append(U[i])
                    test_T.append(T[i])

                else:
                    train_X.append(X[i])
                    train_Y.append(Y[i])
                    train_U.append(U[i])
                    train_T.append(T[i])

            Q_inv = np.eye(train_X[0].shape[1])
            A, g, B = elastic_net_clv(train_X, train_U, train_T, Q_inv, alpha, r_A, r_g, r_B, tol=1)
            sqr_err += compute_prediction_error(test_X, test_Y, test_U, test_T, A, g, B)

        if sqr_err < best_sqr_err:
            best_r = (alpha, r_A, r_g, r_B)
            best_sqr_err = sqr_err
            #print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    np.set_printoptions(suppress=False)
    return best_r


def predict(x0, u, times, A, g, B):
    xt = x0
    zt  = np.concatenate((xt, np.array([0])))
    pt  = np.exp(zt - logsumexp(zt))
    y_pred = np.zeros((times.shape[0], x0.size+1))
    for i in range(1,times.shape[0]):
        dt = times[i] - times[i-1]
        xt = xt + dt*(g + A.dot(pt) + B.dot(u[i-1]))
        zt  = np.concatenate((xt, np.array([0])))
        pt  = np.exp(zt - logsumexp(zt))
        y_pred[i] = pt        
    return y_pred


def compute_prediction_error(X, Y, U, T, A, g, B):
    def compute_total_square_error(y, y_pred):
        err = 0
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err += np.square(yt/yt.sum() - ypt).sum()
        return err
    err = 0
    for x, y, u, t in zip(X, Y, U, T):
        y_pred = predict(x[0], u, t, A, g, B)
        err += compute_total_square_error(y, y_pred)
    return err
