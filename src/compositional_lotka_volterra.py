import numpy as np
import sys
from scipy.special import logsumexp
from scipy.stats import linregress

from src.find_stable_subset import find_stable_subset

class CompositionalLotkaVolterra:
    """Inference for compositional Lotka-Volterra.
    """

    def __init__(self, P=None, T=None, U=None):
        """
        Parameters
        ----------
            P : A list of T_x by D dimensional numpy arrays of
                estimated relative abundances.
            Y  : A list of T_y by D dimensional numpy arrays of
                 time-series observations. T_y denotes the number
                 of observations for sequence y.
            T : A list of T_x by 1 dimensional numpy arrays giving
                the times of each observation x.
            U : An optional list of T_x by P numpy arrays of external
                perturbations for each x.
        """
        self.P = P
        self.T = T

        if P is not None:
            self.X, self.denom_ids = self.construct_log_ratios(P)
        else:
            self.X = None

        if U is None and self.X is not None:
            self.U = [ np.zeros((x.shape[0], 1)) for x in X ]
        else:
            self.U = U

        # Parameter estimates
        self.A = None
        self.g = None
        self.B = None
        self.Q_inv = np.eye(self.P[0].shape[1]) if P is not None else None

        # Regularization parameters
        self.alpha = None
        self.r_A = None
        self.r_g = None
        self.r_B = None


    def construct_log_ratios(self, P, denom_ids = None):
        if denom_ids is None:
            denom_ids = find_stable_subset(P)
        X = []
        for p in P:
            x = np.zeros((p.shape[0], p.shape[1]))
            for t in range(p.shape[0]):
                pt = p[t]
                denom1 = np.sum(pt[denom_ids])
                denom2 = 1-denom1
                denom = np.array([denom1 if i not in denom_ids else denom2 for i in range(p.shape[1])]).flatten()
                xt = np.log(pt) - np.log(denom)
                x[t] = xt
            X.append(x)
        return X, denom_ids


    def get_regularizers(self):
        return self.alpha, self.r_A, self.r_g, self.r_B


    def set_regularizers(self, alpha, r_A, r_g, r_B):
        self.alpha = alpha
        self.r_A = r_A
        self.r_g = r_g
        self.r_B = r_B


    def train(self, verbose=False):
        """Estimate regularization parameters and CLV model parameters.
        """
        if self.alpha is None or self.r_A is None or self.r_g is None or self.r_B is None:
            if verbose:
                print("Estimating regularizers...")
            self.alpha, self.r_A, self.r_g, self.r_B = estimate_elastic_net_regularizers_cv(self.X, self.P, self.U, self.T, self.denom_ids, verbose=verbose)
        
        if verbose:
            print("Estimating model parameters...")
        self.A, self.g, self.B = elastic_net_clv(self.X, self.P, self.U, self.T, self.Q_inv, self.alpha, self.r_A, self.r_g, self.r_B)
        
        if verbose:
            print()


    def predict(self, p, times, u = None):
        """Predict relative abundances one step at time.

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
            u = np.zeros((p.shape[0], 1))

        X, denom_ids = self.construct_log_ratios([p], self.denom_ids)
        x = X[0]

        p_pred = np.zeros((times.shape[0], x[0].size))
        pt = p[0]
        xt = x[0]
        for i in range(1,times.shape[0]):
            dt = times[i] - times[i-1]
            xt = xt + dt*(self.g + self.A.dot(pt) + self.B.dot(u[i-1]))
            pt = compute_rel_abun(xt, self.denom_ids).flatten()
            p_pred[i] = pt
        return p_pred


    def predict_one_step(self, p, times, u = None):
        """Predict relative abundances one step at time.

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
        X, denom_ids = self.construct_log_ratios([p], self.denom_ids)
        x = X[0]

        if u is None:
            u = np.zeros((x.shape[0], 1))

        p_pred = np.zeros((times.shape[0], x[0].size))
        pt = p[0]
        xt = x[0]
        for i in range(1,times.shape[0]):
            dt = times[i] - times[i-1]
            xt = x[i-1] + dt*(g + A.dot(p[i-1]) + B.dot(u[i-1]))
            pt = compute_rel_abun(xt, denom_ids).flatten()
            p_pred[i] = pt
            
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


def elastic_net_clv(X, P, U, T, Q_inv, r_A, r_g, r_B, alpha=1, tol=1e-3, verbose=False, max_iter=100000):

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


    def stack_observations(X, P, U, T):
        # number of observations by xDim
        x_stacked = None
        # number of observations by yDim + 1 + uDim
        pgu_stacked = None
        for x, p, u, times in zip(X, P, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                pt0 = p[t-1]
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
    yDim = xDim
    uDim = U[0].shape[1]
    AgB = np.zeros(( xDim, yDim + 1 + uDim ))

    x_stacked, pgu_stacked = stack_observations(X, P, U, T)
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

        if verbose:# and it % 100 == 0:
            print("\t", it, obj)

        if it > max_iter:
            print("Warning: maximum number of iterations ({}) reached".format(max_iter), file=sys.stderr)
            break

    A = AgB[:,:yDim]
    g = AgB[:,yDim:(yDim+1)].flatten()
    B = AgB[:,(yDim+1):]

    return A, g, B



def estimate_elastic_net_regularizers_cv(X, P, U, T, denom_ids, folds=10, verbose=False):
    if len(X) == 1:
        print("Error: cannot estimate regularization parameters from single sample", file=sys.stderr)
        exit(1)
    elif len(X) < 10:
        folds = len(X)
    
    rs = [0.1, 0.5, 0.90, .95, 1]
    alphas = [0.1, 0.5, 1, 10]
    alpha_rA_rg_rB = []

    for alpha in alphas:
        for r_Ag in rs:
            for r_B in rs:
                alpha_rA_rg_rB.append((alpha, r_Ag, r_Ag, r_B))
    
    np.set_printoptions(suppress=True)
    best_r = 0
    best_sqr_err = np.inf
    for i, aAgB in enumerate(alpha_rA_rg_rB):
        alpha, r_A, r_g, r_B = aAgB
        print("\tTesting regularization parameter set", i+1, "of", len(alpha_rA_rg_rB), file=sys.stderr)
        sqr_err = 0
        for fold in range(folds):
            train_X = []
            train_P = []
            train_U = []
            train_T = []

            test_X = []
            test_P = []
            test_U = []
            test_T = []
            for i in range(len(X)):
                if i % folds == fold:
                    test_X.append(X[i])
                    test_P.append(P[i])
                    test_U.append(U[i])
                    test_T.append(T[i])

                else:
                    train_X.append(X[i])
                    train_P.append(P[i])
                    train_U.append(U[i])
                    train_T.append(T[i])

            Q_inv = np.eye(train_X[0].shape[1])
            A, g, B = elastic_net_clv(train_X, train_P, train_U, train_T, Q_inv, alpha, r_A, r_g, r_B, tol=0.01)
            sqr_err += compute_prediction_error(test_X, test_P, test_U, test_T, A, g, B, denom_ids)

        if sqr_err < best_sqr_err:
            best_r = (alpha, r_A, r_g, r_B)
            best_sqr_err = sqr_err
            #print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    np.set_printoptions(suppress=False)
    return best_r


def compute_rel_abun(x, denom_ids):
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    alt_denom = [i for i in range(x.shape[1]) if i not in denom_ids]
    x_d1 = np.hstack( (x[:,denom_ids], np.zeros((x.shape[0], 1))) )
    x_d2 = np.hstack( (x[:,alt_denom], np.zeros((x.shape[0], 1))) )
    p = np.zeros(x.shape)
    p[:,denom_ids] = np.exp(x[:,denom_ids] - logsumexp(x_d1,axis=1,keepdims=True))
    p[:,alt_denom] = np.exp(x[:,alt_denom] - logsumexp(x_d2,axis=1,keepdims=True))
    p /= p.sum()
    #assert np.allclose(p.sum(), 1), p.sum()
    return p


def predict(x, p, u, times, A, g, B, denom_ids):
    """One step prediction.
    """
    p_pred = np.zeros((times.shape[0], x[0].size))
    pt = p[0]
    xt = x[0]
    for i in range(1,times.shape[0]):
        dt = times[i] - times[i-1]
        xt = x[i-1] + dt*(g + A.dot(p[i-1]) + B.dot(u[i-1]))
        pt = compute_rel_abun(xt, denom_ids).flatten()
        p_pred[i] = pt
    return p_pred


def compute_prediction_error(X, P, U, T, A, g, B, denom_ids):
    def compute_err(p, p_pred):
        """Define error to be 1-r2 per taxon.
        """
        err = 0
        ntaxa = p.shape[1]
        for i in range(ntaxa):
            err += 1-np.square(linregress(p[1:,i],p_pred[1:,i])[2])
        return err/ntaxa
    err = 0
    for x, p, u, t in zip(X, P, U, T):
        p_pred = predict(x, p, u, t, A, g, B, denom_ids)
        err += compute_err(p, p_pred)
    return err/len(X)
