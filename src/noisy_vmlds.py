import numpy as np
import scipy.optimize
import sys
from scipy.stats import norm, multinomial
from scipy.special import gammaln, xlogy, logsumexp

import time

if __name__ == "__main__":
    from blk_tridiag_inv import compute_blk_tridiag, compute_blk_tridiag_inv_b
else:
    from src.blk_tridiag_inv import compute_blk_tridiag, compute_blk_tridiag_inv_b


def multinomial(x, n, p):
    return gammaln(n+1) + np.sum(xlogy(x,p) - gammaln(x+1), axis=-1)


def block_multiply(AA, BB, x = None):
    """Block multiply x by symmetric tridiagonal matrix with
    diagonal entries AA and upper diagonal entries BB.
    """
    # means no BB specified
    if x is None: 
        x = BB
        res = np.zeros(x.shape)
        for t in range(AA.shape[0]):
            res[t] = AA[t].dot(x[t])
        return res
    else:
        res = np.zeros(x.shape)
        tpts = AA.shape[0]

        res[0] = AA[0].dot(x[0]) + BB[0].dot(x[1])
        for t in range(1, tpts-1):
            res[t] = BB[t].T.dot(x[t-1]) + AA[t].dot(x[t]) + BB[t].dot(x[t+1])
        res[tpts-1] = BB[tpts-2].T.dot(x[tpts-2]) + AA[tpts-1].dot(x[tpts-1])
        return res


def compute_blk_inner_prod(x, lambda_AA, lambda_BB = None):
    if lambda_BB is None:
        lambdaX = block_multiply(lambda_AA, x)
    else:
        lambdaX = block_multiply(lambda_AA, lambda_BB, x)
    return (lambdaX*x).sum()


def compute_blk_log_det(AA, BB):
    def combine_blk_diag(AA, BB):
        T = AA.shape[0]
        dim = AA.shape[1]
        stacked = np.zeros((dim*T, dim*T))
        stacked[:dim,:dim] = AA[0]
        stacked[:dim,dim:2*dim] = BB[0]
        for t in range(1, T-1):
            stacked[dim*t:dim*(t+1),dim*t:dim*(t+1)] = AA[t]
            stacked[dim*t:dim*(t+1),dim*(t+1):dim*(t+2)] = BB[t]
            stacked[dim*t:dim*(t+1),dim*(t-1):dim*t] = BB[t].T
        stacked[dim*(T-1):,dim*(T-1):] = AA[T-1]
        stacked[dim*(T-1):,dim*(T-2):dim*(T-1)] = BB[T-2].T
        return stacked

    stacked = combine_blk_diag(AA, BB)
    sgn_log_det = np.linalg.slogdet(stacked)
    assert sgn_log_det[0] == 1
    return sgn_log_det[1]


def multiply_across_axis(AA, x):
    res = np.zeros(AA.shape)
    for t in range(res.shape[0]):
        res[t] = AA[t]*x[t]
    return res


def compute_condition_number(AA, BB, AA_inv, BB_inv):
    def combine_block_diag(AA, BB):
        T = AA.shape[0]
        dim = AA.shape[1]
        stacked = np.zeros((dim*T, dim*T))
        stacked[:dim,:dim] = AA[0]
        stacked[:dim,dim:2*dim] = BB[0]
        for t in range(1, T-1):
            stacked[dim*t:dim*(t+1),dim*t:dim*(t+1)] = AA[t]
            stacked[dim*t:dim*(t+1),dim*(t+1):dim*(t+2)] = BB[t]
            stacked[dim*t:dim*(t+1),dim*(t-1):dim*t] = BB[t].T
        stacked[dim*(T-1):,dim*(T-1):] = AA[T-1]
        stacked[dim*(T-1):,dim*(T-2):dim*(T-1)] = BB[T-2].T
        return stacked
    AB = combine_block_diag(AA, BB)
    AB_inv = combine_block_diag(AA_inv, BB_inv)
    w_AB, v = np.linalg.eig(AB)
    w_AB_inv, v = np.linalg.eig(AB_inv)
    l1 = np.max(np.abs(w_AB))
    l2 = np.max(np.abs(w_AB_inv))

    u, s, vh = np.linalg.svd(AB)
    return l1*l2


class NoisyVMLDS:
    """Variational Inference for a Noisy Multinomial Linear
    Dynamical System.
    """

    def __init__(self, Y, U, T, denom):
        """
        Parameters
        ----------
            Y  : A list of T_y by D dimensional numpy arrays of
                 time-series observations. T_y denotes the number
                 of observations for sequence y.
            U  : A list of T_y by P dimensional numpy arrays giving
                 external perturbations.
            T  : A list of T_y by 1 dimensional numpy arrays giving
                 the times of each observation in a sequence y.
            denom : denominator of the additive log-ratio transformation
        """
        for t in T:
            if t.size <= 1:
                print("Error: sequence has 1 or fewer time points", file=sys.stderr)
                exit(1)
        for y in Y:
            totals = y.sum(axis=1)
            if np.any(np.allclose(totals, 1)):
                print("Error: attempting to run estimation step on relative abundances (read counts are required)", file=sys.stderr)
                exit(1)

        self.Y = self.swap_last_taxon(Y, denom) # last taxon is denominator
        self.V = self.parse_perturbations(U)
        self.T = T
        self.denom = denom
        self.obs_dim = Y[0].shape[1]
        self.latent_dim = self.obs_dim - 1

        # posterior means, the variational parameters
        self.X = [ np.zeros( (y.shape[0], self.latent_dim) ) for y in Y ]
        # noisy observations, variational parameters 
        self.Z = [ np.zeros( (y.shape[0], self.latent_dim) ) for y in Y ]
        # expectations for zeros
        self.W = [ np.ones( (y.shape[0], self.obs_dim) ) for y in Y ]
        # pairwise expectations for zeros
        self.W0_W1 = [ 0.5*np.ones( (y.shape[1], y.shape[0], 2, 2) ) for y in Y]

        for i,y in enumerate(self.Y):
            y = np.copy(y)
            self.W[i][y > 0] = 1
            self.W[i][y == 0] = 0.5

            y += 1
            self.X[i] = (np.log(y[:,:-1]).T - np.log(y[:,-1])).T
            self.Z[i] = (np.log(y[:,:-1]).T - np.log(y[:,-1])).T

        # initial state space variance
        self.sigma2_0 = 5*np.eye(self.latent_dim)
        # state space variance
        self.sigma2 = 0.2*np.eye(self.latent_dim)
        # perturbation variance
        self.sigma2_p = np.eye(self.latent_dim)
        # observation variance
        self.gamma2 = np.ones(self.latent_dim)
        # transitions for zeros
        self.A, self.A_init = self.initialize_A(Y)

        # variance of X, block precision and block covariance
        self.lambda_AA = [ [] for y in Y ] # diagonal blocks of the precision matrix
        self.lambda_BB = [ [] for y in Y ] # upper diagonal blocks of the precision matrix
        self.sigma_AA = [ [] for y in Y ] # diagonal blocks of the covariance matrix
        self.sigma_BB = [ [] for y in Y ] # upper diagonal blocks of the covariance matrix
        self.sigma_S = [ [] for y in Y ] # intermediate computation used to invert lambda
        self.gamma_inv_AA = [ [] for y in Y ] # diagonal blocks for noisy z covariance
        
        # initialize parameters
        self.update_variance()


    def swap_last_taxon(self, Y, denom):
        Y_swapped = []
        for y in Y:
            y = np.copy(y)
            tmp = np.copy(y[:,-1])
            y[:,-1] = np.copy(y[:,denom])
            y[:,denom] = tmp
            Y_swapped.append(y)
        return Y_swapped


    def get_relative_abundances(self, X=None, Y=None):
        P = []
        if X is None:
            X = self.X
            Y = self.Y

        denom = self.denom
        var_Z = np.ones(self.gamma2.shape)
        for x,y in zip(X, Y):
            x1 = np.hstack((x + 0.5*var_Z, np.zeros((x.shape[0], 1))))
            p = np.exp(x1 - logsumexp(x1, axis=1, keepdims=True))

            # taxa without any observed counts (zero rows)
            # should remain fixed at 0.
            p[:,y.sum(axis=0) == 0] = 0
            p /= p.sum(axis=1, keepdims=True)

            # place in same order as original input
            tmp = np.copy(p[:,-1])
            p[:,-1] = np.copy(p[:,denom])
            p[:,denom] = tmp
            P.append(p)
        return P


    def get_posterior_nonzero_probs(self, W=None):
        if W is None:
            W = self.W

        W_swapped = []
        for w in self.W:
            w = np.copy(w)
            tmp = np.copy(w[:,-1])
            w[:,-1] = np.copy(w[:,self.denom])
            w[:,self.denom] = tmp
            W_swapped.append(w)
        return W_swapped


    def parse_perturbations(self, U):
        # U gives perturbations at each time point,
        # but we only need to adjust the variance for
        # the first and last
        V = [] # 1 if time point uses sigma2_p, 0 otherwise
        for u in U:
            in_perturb = False
            v = []
            for i,ut in enumerate(u):
                if np.any(ut) > 0 and not in_perturb:
                    v.append(1)
                    in_perturb = True
                elif np.any(ut) > 0 and in_perturb:
                    v.append(0)
                elif np.all(ut) == 0 and in_perturb:
                    i = 1 if v[i-1] == 0 else 0
                    v.append(i)
                    in_perturb = False
                else:
                    v.append(0)
            v = np.array(v)
            V.append(v)
        return V


    def initialize_A(self, Y):
        A = np.zeros((self.obs_dim, 2, 2))
        A_init = np.zeros(Y[0].shape[1])
        for y in Y:
            y0 = y[:-1]
            y1 = y[1:]
            A[:,1,1] += np.logical_and(y0 != 0, y1 != 0).sum()
            A[:,0,1] += np.logical_and(y0 == 0, y1 != 0).sum()
            A[:,1,0] += np.logical_and(y0 != 0, y1 == 0).sum()
            A[:,0,0] += np.logical_and(y0 == 0, y1 == 0).sum()

            A_init += (y[0] != 0).sum()

        A[A == 0] = 1e-8

        denom0 = A[:,0,1] + A[:,0,0]
        denom1 = A[:,1,1] + A[:,1,0]
        A[:,1,1] /= denom1
        A[:,1,0] /= denom1
        A[:,0,1] /= denom0
        A[:,0,0] /= denom0
        A[A == 0] = 1e-4
        A[A == 1] = 1-1e-4

        A_init /= (len(Y)*Y[0].shape[1])
        A_init[A_init == 0] = 1e-4
        A_init[A_init == 1] = 1-1e-4
        return A, A_init


    def optimize(self, verbose=False):
        """Run the VI algorithm.
        """
        prv = -np.inf
        # this is not guaranteed to be strictly increasing
        nxt = self.compute_elbo()
        it = 0

        X_prv = None
        Z_prv = None

        converged = False

        while not converged:
        #while np.abs(prv - nxt) > 0.1:
            # if verbose:
            #     # print("\tit:", it, "delta:", np.abs(nxt - prv))
            #     print("\tit:", it, "delta:", np.max((x_diff, z_diff)))

            X_prv = [np.copy(x) for x in self.X]
            Z_prv = [np.copy(z) for z in self.Z]
            prv_inner = -np.inf
            nxt_inner = nxt
            it_inner = 0
            while np.abs(prv_inner - nxt_inner) > 1:
                self.update_Z()
                self.update_X()
                self.update_W()
                prv_inner = nxt_inner
                nxt_inner = self.compute_elbo()
                it_inner += 1

            self.update_A()
            if it < 2:
                self.update_sigmas()
                self.update_gamma()
                self.update_variance()

            x_diff, z_diff = self.converged(self.X, X_prv, self.Z, Z_prv)
            converged = x_diff < 0.001 and z_diff < 0.001

            prv = nxt
            nxt = self.compute_elbo()
            it += 1

            if verbose:
                # print("\tit:", it, "delta:", np.abs(nxt - prv))
                print("\tit:", it, "delta:", np.max((x_diff, z_diff)))


    def converged(self, X, X_prv, Z, Z_prv):
        if X_prv is None or Z_prv is None:
            return np.inf, np.inf
        x_diff = 0.
        z_diff = 0.
        total = 0
        for x,x_prv,z,z_prv in zip(X, X_prv, Z, Z_prv):
            x_max = np.max(np.sqrt(np.square(x-x_prv)))
            if x_max > x_diff:
                x_diff = x_max
            z_max = np.max(np.sqrt(np.square(z-z_prv)))
            if z_max > z_diff:
                z_diff = z_max
        threshold = 1e-3
        return x_diff, z_diff


    def get_latent_means(self):
        X = [ np.copy(x) for x in self.X ]
        return X


    def compute_elbo(self):
        """Compute the variational objective function. Note the entropy of q(w)
           is not included in this function, and therefore the elbo is not guaranteed
           to increase each iteration.
        """
        def compute_blk_trace(AA_1, AA_2):
            tr = 0
            for D1, D2 in zip(AA_1, AA_2):
                tr += np.trace(np.dot(D1, D2))
            return tr

        lat_dim = self.latent_dim
        elbo = 0
        for i in range(len(self.X)):
            lambda_AA = self.lambda_AA[i]
            lambda_BB = self.lambda_BB[i]
            gamma_inv_AA = self.gamma_inv_AA[i]
            var_Z = self.gamma2
            w = self.W[i]
            x = self.X[i]
            y = self.Y[i]
            z = self.Z[i]
            v = self.V[i]
            w0_w1 = self.W0_W1[i]
            tpts = lambda_AA.shape[0]

            # state space
            state_space = -0.5*(compute_blk_inner_prod(x, lambda_AA, lambda_BB)
                                + compute_blk_inner_prod(z-x, multiply_across_axis(gamma_inv_AA, w[:,:lat_dim]))
                                + (np.log(2*np.pi*var_Z)*w[:,:lat_dim]).sum())
            #state_space += -0.5*np.linalg.slogdet(self.sigma2)[1]*np.abs(1 - v[1:]).sum()
            #state_space += -0.5*np.linalg.slogdet(self.sigma2_p)[1]*v[1:].sum()
            #state_space += -0.5*np.linalg.slogdet(self.sigma2_0)[1]

            # observations
            y = self.Y[i]
            np.seterr(divide="ignore") # log of 0 is handled appropriately here
            p = np.hstack([np.log(w[:,:lat_dim]) + z + var_Z, np.expand_dims(np.log(w[:,lat_dim]),axis=1)])
            np.seterr(divide="warn")
            p = np.exp(p - logsumexp(p,axis=1,keepdims=True))
            p /= p.sum(axis=1,keepdims=True)
            observations = multinomial(y, y.sum(axis=1), p).sum()

            # # zeros
            A = self.A
            A_init = self.A_init
            w0_w1 = self.W0_W1[i]
            try:
                np.seterr(divide="raise")
                zeros = ( w0_w1[:,1:,1,1].T*np.log(A[:,1,1]) +
                          w0_w1[:,1:,1,0].T*np.log(A[:,1,0]) +
                          w0_w1[:,1:,0,1].T*np.log(A[:,0,1]) +
                          w0_w1[:,1:,0,0].T*np.log(A[:,0,0])
                         ).sum() + \
                          (w[0,:]*np.log(A_init) + (1-w[0,:])*np.log((1-A_init))).sum()
                np.seterr(divide="warn")
            except FloatingPointError as e:
                print(e, file=sys.stderr)
                exit(1)

            elbo += state_space + zeros + observations
        return elbo


    def update_X(self):
        """Compute the optimal posterior means.
        """
        lat_dim = self.latent_dim
        prv = self.compute_elbo()
        for i in range(len(self.X)):
            z = self.Z[i]
            w = self.W[i]
            x_prv = self.X[i]
            lambda_AA = self.lambda_AA[i]
            lambda_BB = self.lambda_BB[i]
            w_gamma_inv_AA = multiply_across_axis(self.gamma_inv_AA[i], w[:,:lat_dim])
            var_Z = self.gamma2
            w_gamma_inv_z = (w[:,:lat_dim]/var_Z)*z
            D, OD, S = compute_blk_tridiag(lambda_AA + w_gamma_inv_AA, lambda_BB)
            x = compute_blk_tridiag_inv_b(S,D,w_gamma_inv_z)
            self.X[i] = x

            if self.compute_elbo() < prv:
                self.X[i] = x_prv

                # if np.max(np.abs(self.X[i])) > 20:
                #     print("bad state space iteration", file=sys.stderr)
                #     exit(1)

    def update_Z(self):
        """Compute the optimal posterior means.
        """
        def block_diag_multiply(AA, z):
            res = np.zeros(z.shape)
            for t in range(AA.shape[0]):
                res[t] = AA[t].dot(z[t])
            return res

        def compute_obj(z, w, x, y, gamma_inv_AA, var_Z):
            """Compute objective function with respect to z = E[Z].
            """
            z = z.reshape(x.shape)
            lat_dim = self.latent_dim
            n = y.sum(axis=1)
            state_space = -0.5*compute_blk_inner_prod(z-x, multiply_across_axis(gamma_inv_AA, w[:,:lat_dim])) 

            #wl = w[:,lat_dim].reshape((w[:,lat_dim].size, 1))
            np.seterr(divide="ignore") # log of 0 is handled appropriately here
            p = np.hstack([np.log(w[:,:lat_dim]) + z + 0.5*var_Z, np.expand_dims(np.log(w[:,lat_dim]),axis=1)])
            np.seterr(divide="warn")
            p = np.exp(p - logsumexp(p,axis=1,keepdims=True))
            p /= p.sum(axis=1,keepdims=True)
            obs = multinomial(y,n,p).sum()

            # obs = (y[:,:lat_dim]*w[:,:lat_dim]*z).sum() \
            #         - (n*np.log(w[:,lat_dim] + (w[:,:lat_dim]*np.exp(z + var_Z)).sum(axis=1))).sum()
            return -(state_space + obs)


        def compute_grad(z, w, x, y, gamma_inv_AA, var_Z):
            """Compute gradient with respect to z = E[Z].
            """
            assert not np.any(np.isnan(z)), z
            z = z.reshape(x.shape)
            n = y.sum(axis=1, keepdims=True)
            lat_dim = self.latent_dim
            np.seterr(divide="ignore") # log of 0 is handled appropriately here
            log_denom = np.hstack([np.log(w[:,:lat_dim]) + z + 0.5*var_Z, np.expand_dims(np.log(w[:,lat_dim]),axis=1)])
            log_denom = logsumexp(log_denom,axis=1)
            log_numer = np.log(n) + np.log(w[:,:lat_dim]) + z + 0.5*var_Z
            #log_numer = np.log(w[:,:lat_dim]) + z + var_Z
            np.seterr(divide="warn")

            blk_grad_z = -block_diag_multiply(multiply_across_axis(gamma_inv_AA, w[:,:lat_dim]), z-x) + \
                            y[:,:lat_dim]*w[:,:lat_dim] - \
                            np.exp(log_numer.T - log_denom).T
                            #np.exp(log_numer.T - np.log(denom)).T

            assert np.all(np.isfinite(blk_grad_z)), str(z) + "\n"  + \
                                                    str(numer) + "\n" + str(denom)
            return -blk_grad_z


        def minimize(z, w, x, y, gamma_inv_AA, var_Z, max_iter=100, verbose=False):
            """Minimize using conjugate gradient.
            """
            prv = np.inf
            nxt = compute_obj(z, w, x, y, gamma_inv_AA, var_Z)
            grad_z = compute_grad(z, w, x, y, gamma_inv_AA, var_Z)
            p = -grad_z

            c1 = 0.0001
            c2 = 0.1
            it = 0
            #while np.sqrt(np.square(grad_z).sum()) > 1:
            while np.abs(prv - nxt) > 1e-3:

                if verbose:
                    print("it:", it, "obj:", nxt)

                if it > max_iter:
                    break

                ss = 0.001
                prv = nxt
                z_prv = np.copy(z)
                z = z_prv + ss*p
                nxt = compute_obj(z, w, x, y, gamma_inv_AA, var_Z)
            
                # Wolfe conditions
                while nxt > prv + c1*ss*(grad_z*p).sum():
                    ss/=2
                    z = z_prv + ss*p
                    nxt = compute_obj(z, w, x, y, gamma_inv_AA, var_Z)
                assert not np.any(np.isnan(z)), str(prv_grad_z) + "\n" + str(grad_z)

                prv_grad_z = np.copy(grad_z)
                grad_z = compute_grad(z, w, x, y, gamma_inv_AA, var_Z)
                b = (grad_z*grad_z).sum() / (prv_grad_z*prv_grad_z).sum()
                p = -grad_z + b*p
                it += 1

                if nxt > prv + 1e-2:
                    print("Warning: increasing objective in Z.\n" +
                          "\twas", prv,
                          "\tis",nxt,
                           file=sys.stderr)
                    exit()

            return z


        for i in range(len(self.X)):
            y = self.Y[i]
            x = self.X[i]
            w = self.W[i]
            z = np.copy(self.Z[i])
            tpts = y.shape[0]
            gamma_inv_AA = self.gamma_inv_AA[i]
            #var_Z = self.gamma2
            var_Z = np.ones(self.gamma2.shape)
            z = minimize(z, w, x, y, gamma_inv_AA, var_Z, verbose=False)
            self.Z[i] = z


    def update_W(self):
        """Using the alpha-gamma algorithm.
        """

        def fwd_pass(w, x, y, z, var_Z):
            w = np.copy(w)
            tpts = w.shape[0]
            ntaxa = w.shape[1]
            A = self.A
            A_init = self.A_init

            # normalized forward probabilities
            alpha = np.zeros(w.shape)
 
            np.seterr(divide="ignore") # zeros are appropriately handled here
            p = np.concatenate((z[0] + var_Z, [0]))
            p = np.tile(p, ntaxa).reshape(ntaxa, ntaxa)
            w0 = np.tile(np.log(w[0]), ntaxa).reshape(ntaxa, ntaxa)
            np.fill_diagonal(w0, 0)
            p += w0
            p = np.exp(p - logsumexp(p, axis=1, keepdims=True))
            n = y[0].sum()
            y0 = np.tile(y[0],ntaxa).reshape(ntaxa,ntaxa)
            alpha_w1 = np.log(A_init) + multinomial(y0,n,p)
            alpha_w1[:ntaxa-1] += norm.logpdf(loc=x[0], scale=np.sqrt(var_Z), x=z[0])

            np.fill_diagonal(p, 0)
            p /= p.sum(axis=1,keepdims=True)
            alpha_w0 = np.zeros(ntaxa)
            alpha_w0[y[0] > 0] = -np.inf
            alpha_w0[y[0] == 0] = np.log(1 - A_init)[y[0] == 0] + multinomial(y0, y0.sum(), p)[y[0] == 0]
            alpha[0] = np.exp(alpha_w1 - logsumexp(np.vstack([alpha_w0, alpha_w1]).T,axis=1))
            assert np.all(alpha[0] >= 0) and np.all(alpha[0] <= 1)

            for t in range(1, tpts):
                alpha_w1 = np.zeros(ntaxa)
                alpha_w0 = np.zeros(ntaxa)
                at0 = alpha[t-1]

                p = np.concatenate((z[t] + var_Z, [0]))
                p = np.tile(p, ntaxa).reshape(ntaxa, ntaxa)
                w0 = np.tile(np.log(w[t]), ntaxa).reshape(ntaxa, ntaxa)
                np.fill_diagonal(w0, 0)
                p += w0
                p = np.exp(p - logsumexp(p, axis=1, keepdims=True))
                n = y[t].sum()
                y0 = np.tile(y[t],ntaxa).reshape(ntaxa,ntaxa)
                alpha_w1 = np.log(A[:,1,1]*at0 + A[:,0,1]*(1-at0)) + multinomial(y0,n,p)
                alpha_w1[:ntaxa-1] += norm.logpdf(loc=x[t], scale=np.sqrt(var_Z), x=z[t])

                np.fill_diagonal(p, 0)
                p /= p.sum(axis=1,keepdims=True)
                alpha_w0 = np.zeros(ntaxa)
                alpha_w0[y[t] > 0] = -np.inf
                alpha_w0[y[t] == 0] = np.log(A[:,1,0]*at0 + A[:,0,0]*(1-at0))[y[t] == 0] + multinomial(y0, n, p)[y[t] == 0]

                np.set_printoptions(threshold=np.inf)
                assert np.all(np.logical_or(np.isfinite(alpha_w0), np.isfinite(alpha_w1)) >= 1), str(p[0]) + "\n" + str(y[t])
                alpha[t] = np.exp(alpha_w1 - logsumexp(np.vstack([alpha_w0, alpha_w1]).T,axis=1))
                assert np.all(alpha[t] >= 0) and np.all(alpha[t] <= 1), alpha[t]

            return alpha

        def bwd_pass(y, alpha):
            # posterior probabilities
            gamma = np.zeros(alpha.shape)
            A = self.A
            tpts = w.shape[0]
            ntaxa = w.shape[1]

            gamma[-1] = alpha[-1]
            np.seterr(divide="ignore")
            for t in range(tpts-2, -1, -1):
                gt1 = gamma[t+1]
                gamma[t,alpha[t] == 1] = 1
                log_p_gamma_w0 = np.log(1-alpha[t]) + np.log(A[:,0,1]*gt1 + A[:,0,0]*(1-gt1))
                log_p_gamma_w1 = np.log(alpha[t]) + np.log(A[:,1,1]*gt1 + A[:,1,0]*(1-gt1))
                gamma[t] = np.exp(log_p_gamma_w1 - logsumexp(np.vstack([log_p_gamma_w1, log_p_gamma_w0]).T,axis=1))
                assert np.all(np.logical_or(np.isfinite(log_p_gamma_w0), np.isfinite(log_p_gamma_w1)) >= 1)
                assert np.all(gamma[t] >= 0) and np.all(gamma[t] <= 1)
            np.seterr(divide="warn")

            return gamma

        def pairwise_pass(w, x, y, z, var_Z, alpha, gamma):
            w = np.copy(w)
            # w_{t-1}, w_t pairwise probabilities
            w0_w1 = np.zeros((w.shape[1], w.shape[0], 2, 2))
            A_init = self.A_init
            A = self.A
            p0 = gamma[0]
            ntaxa = w.shape[1]
            tpts = w.shape[0]

            # log of invalid values are dealt with automatically
            # so let's turn off the warnings here. the assertions
            # should catch any unexpected errors
            np.seterr(divide="ignore", invalid="ignore")

            for t in range(1, tpts):
                at0 = alpha[t-1]
                at1 = alpha[t]
                gt1 = gamma[t]


                p = np.concatenate((z[t] + var_Z, [0]))
                p = np.tile(p, ntaxa).reshape(ntaxa, ntaxa)
                w0 = np.tile(np.log(w[t]), ntaxa).reshape(ntaxa, ntaxa)
                np.fill_diagonal(w0, 0)
                p += w0
                p = np.exp(p - logsumexp(p, axis=1, keepdims=True))
                n = y[t].sum()
                y0 = np.tile(y[t],ntaxa).reshape(ntaxa,ntaxa)
                obs_1 = multinomial(y0,n,p)
                obs_1[:ntaxa-1] += norm.logpdf(loc=x[t], scale=np.sqrt(var_Z), x=z[t])

                np.fill_diagonal(p, 0)
                p /= p.sum(axis=1,keepdims=True)
                obs_0 = np.zeros(ntaxa)
                obs_0[y[t] > 0] = -np.inf
                obs_0[y[t] == 0] = multinomial(y0,n,p)[y[t] == 0]

                ids = np.logical_and(y[t] > 0, y[t-1] > 0)
                if np.sum(ids) > 0:
                    w0_w1[ids,t,0,1] = 0
                    w0_w1[ids,t,0,0] = 0
                    w0_w1[ids,t,1,1] = 1
                    w0_w1[ids,t,1,0] = 0

                ids = np.logical_and(y[t] > 0, y[t-1] == 0)
                if np.sum(ids) > 0:
                    w0_w1[ids,t,0,1] = np.log(1 - at0[ids]) + obs_1[ids] + np.log(gt1[ids]) + np.log(A[ids,0,1]) - np.log(at1[ids])
                    w0_w1[ids,t,0,0] = -np.inf
                    w0_w1[ids,t,1,1] = np.log(at0[ids]) + obs_1[ids] + np.log(gt1[ids]) + np.log(A[ids,1,1]) - np.log(at1[ids])
                    w0_w1[ids,t,1,0] = -np.inf
                    log_denom = logsumexp(w0_w1[ids,t],axis=(1,2),keepdims=True)
                    w0_w1[ids,t] = np.exp(w0_w1[ids,t] - log_denom)
                    assert np.all(np.abs(w0_w1[ids,t].sum(axis=(1,2)) - 1) < 1e-2), w0_w1[ids,t]

                ids = np.logical_and(y[t] == 0, y[t-1] > 0)
                if np.sum(ids) > 0:
                    w0_w1[ids,t,0,1] = -np.inf
                    w0_w1[ids,t,0,0] = -np.inf
                    w0_w1[ids,t,1,1] = np.log(at0[ids]) + obs_1[ids] + np.log(gt1[ids]) + np.log(A[ids,1,1]) - np.log(at1[ids])
                    w0_w1[ids,t,1,0] = np.log(at0[ids]) + obs_0[ids] + np.log(1-gt1[ids]) + np.log(A[ids,1,0]) - np.log(1-at1[ids])
                    w0_w1[np.logical_and(ids,gt1==0),t,1,1] = -np.inf
                    w0_w1[np.logical_and(ids,gt1==1),t,1,0] = -np.inf
                    log_denom = logsumexp(w0_w1[ids,t],axis=(1,2),keepdims=True)
                    w0_w1[ids,t] = np.exp(w0_w1[ids,t] - log_denom)
                    assert np.all(np.abs(w0_w1[ids,t].sum(axis=(1,2)) - 1) < 1e-2), str(w0_w1[ids,t]) + "\n" + str(log_denom)

                ids = np.logical_and(y[t] == 0, y[t-1] == 0)
                if np.sum(ids) > 0:
                    w0_w1[ids,t,0,1] = np.log(1 - at0[ids]) + obs_1[ids] + np.log(gt1[ids]) + np.log(A[ids,0,1]) - np.log(at1[ids])
                    w0_w1[ids,t,0,0] = np.log(1 - at0[ids]) + obs_0[ids] + np.log(1-gt1[ids]) + np.log(A[ids,0,0]) - np.log(1-at1[ids])
                    w0_w1[ids,t,1,1] = np.log(at0[ids]) + obs_1[ids] + np.log(gt1[ids]) + np.log(A[ids,1,1]) - np.log(at1[ids])
                    w0_w1[ids,t,1,0] = np.log(at0[ids]) + obs_0[ids] + np.log(1-gt1[ids]) + np.log(A[ids,1,0]) - np.log(1-at1[ids])

                    w0_w1[np.logical_and(ids, gt1==0),t,0,1] = -np.inf
                    w0_w1[np.logical_and(ids, gt1==0),t,1,1] = -np.inf
                    w0_w1[np.logical_and(ids, gt1==1),t,0,0] = -np.inf
                    w0_w1[np.logical_and(ids, gt1==1),t,1,0] = -np.inf

                    log_denom = logsumexp(w0_w1[ids,t],axis=(1,2),keepdims=True)
                    w0_w1[ids,t] = np.exp(w0_w1[ids,t] - log_denom)
                    assert np.all(np.abs(w0_w1[ids,t].sum(axis=(1,2)) - 1) < 1e-2), str(w0_w1[ids,t]) + "\n" + str(log_denom)

                assert np.all(w0_w1[:,t,0,1] >= 0) and np.all(w0_w1[:,t,0,1] <= 1)
                assert np.all(w0_w1[:,t,1,1] >= 0) and np.all(w0_w1[:,t,1,1] <= 1)

            # reset error messages
            np.seterr(divide="warn", invalid="warn")
            return w0_w1

        def post(w, x, y, z, var_Z):
            w_nxt = np.copy(w)
            w0_w1_nxt = np.zeros((w.shape[1], w.shape[0], 2, 2))
            ntaxa = w.shape[1]
            log_p_zy = 0
            alpha = fwd_pass(w, x, y, z, var_Z)
            gamma = bwd_pass(y, alpha)
            w_nxt = gamma
            w0_w1_nxt = pairwise_pass(w_nxt, x, y, z, var_Z, alpha, gamma)
            return w_nxt, w0_w1_nxt

        def optimize(w, w0_w1, x, y, z, gamma_inv_AA, var_Z, verbose=False):
            prv_w = np.copy(w)
            prv_w0_w1 = np.copy(w0_w1)
            w, w0_w1 = post(w, x, y, z, var_Z)

            while not np.allclose(prv_w, w) and not np.allclose(prv_w0_w1, w0_w1):
                prv_w = np.copy(w)
                prv_w0_w1v = np.copy(w0_w1)
                w, w0_w1 = post(w, x, y, z, var_Z)

            return w, w0_w1

        for i in range(len(self.X)):
            y = self.Y[i]
            x = self.X[i]
            z = self.Z[i]
            w = np.copy(self.W[i])
            w0_w1 = np.copy(self.W0_W1[i])
            gamma_inv_AA = self.gamma_inv_AA[i]
            #var_Z = self.gamma2
            var_Z = np.ones(self.gamma2.shape)

            w, w0_w1 = optimize(w, w0_w1, x, y, z, gamma_inv_AA, var_Z, verbose=False)
            self.W[i] = w
            self.W0_W1[i] = w0_w1

    def update_A(self):
        A = np.zeros(self.A.shape)
        A_init1 = np.zeros(self.A_init.shape)
        A_init0 = np.zeros(self.A_init.shape)

        for i,w in enumerate(self.W):
            tpts = w.shape[0]
            w0_w1 = self.W0_W1[i]

            A[:,1,1] += (w0_w1[:,1:,1,1]).sum(axis=1)
            A[:,0,1] += (w0_w1[:,1:,0,1]).sum(axis=1)
            A[:,1,0] += (w0_w1[:,1:,1,0]).sum(axis=1)
            A[:,0,0] += (w0_w1[:,1:,0,0]).sum(axis=1)

            A_init1 += w[0,:]
            A_init0 += (1-w[0,:])
        
        A[A == 0] = 1e-8
        denom0 = A[:,0,1] + A[:,0,0]
        denom1 = A[:,1,1] + A[:,1,0]
        A[:,1,1] /= denom1
        A[:,1,0] /= denom1
        A[:,0,1] /= denom0
        A[:,0,0] /= denom0
        A[A == 0] = 1e-8
        A[A == 1] = 1-1e-8

        A_init = A_init1 / (A_init1 + A_init0)
        A_init[A_init == 0] = 1e-8
        A_init[A_init == 1] = 1-1e-8
        self.A = A
        self.A_init = A_init


    def update_sigmas(self):
        """Update the state space variance parameters.
        """
        sigma2_0 = 0
        sigma2 = 0
        sigma2_p = 0

        Mt = 0
        St = 0
        Mt_p = 0
        St_p = 0

        n_perturb = 0
        n_normal = 0

        for i in range(len(self.X)):
            x = self.X[i]
            v = self.V[i]
            sigma2_0 += np.outer(x[0], x[0])
            tpts = x.shape[0]

            for t in range(1, tpts):
                dt = self.T[i][t] - self.T[i][t-1]

                if v[t] == 1:
                    tmp = (x[t] - x[t-1]) / np.sqrt(dt)
                    St_p = St_p + tmp * tmp
                    n_perturb += 1
                else:
                    tmp = (x[t] - x[t-1]) / np.sqrt(dt)
                    St = St + np.outer(tmp, tmp)
                    n_normal += 1

        sigma2_0 /= len(self.X)
        sigma2 = St / np.max((n_normal, 1))
        sigma2_p = St_p / np.max((n_normal, 1))

        # Numerical errors can cause the covariance matrix
        # to have nonpositive and nonreal eigenvalues and
        # eigenvectors. Below ensures that we end up with
        # a positive definite matrix. Clipping the eigenvalues
        # bounds the condition number, a measure of stability.
        w,v = np.linalg.eig(sigma2)
        v = np.real(v)
        w = np.real(w)
        w = np.clip(w, 1e-3, 5)
        sigma2 = v.dot(np.diag(w)).dot(v.T)
        self.sigma2 = sigma2

        self.sigma2_p = np.clip(sigma2_p, 1e-3, 5)*np.eye(self.latent_dim)


    def update_gamma(self):
        lat_dim = self.latent_dim
        Mt = np.zeros(lat_dim)
        St = np.zeros(lat_dim)
        total = 0
        for i in range(len(self.X)):
            x = self.X[i]
            z = self.Z[i]
            w = self.W[i]
            tpts = x.shape[0]
            for t in range(tpts):
                tmp = np.sqrt(w[t,:lat_dim])*(z[t]-x[t])
                St = St + tmp*tmp
                total += 1

        gamma2 = St / (total)
        gamma2 = np.clip(gamma2, 1e-3, 5)
        self.gamma2 = gamma2


    def update_variance(self):
        """Compute the block precision matrix, block covariance matrix, and
        variance of each component of X.
        """
        gamma_inv_AA = 1/self.gamma2 * np.eye(self.latent_dim)
        for i in range(len(self.X)):
            tpts = self.X[i].shape[0]
            self.lambda_AA[i], self.lambda_BB[i] = self.compute_lambda_blk_tridiag(self.T[i], self.V[i])
            self.gamma_inv_AA[i] = np.array([gamma_inv_AA for t in range(tpts)])


    def compute_lambda_blk_tridiag(self, times, v):
        """Compute the symmetric block tridiagonal precision matrix
        for the state space.

        Parameters
        ----------
            times : a matrix from self.T

        Returns
        _______
            AA : block diagonal entries of the precision matrix
            BB : block upper-diagonal entries of the precision matrix
        """
        s2_0_inv = np.linalg.pinv(self.sigma2_0)
        s2_p_inv = np.linalg.pinv(self.sigma2_p)
        s2_inv = np.linalg.pinv(self.sigma2)
        lat_dim = self.latent_dim
        t_pts = times.shape[0]

        assert np.all(np.diag(s2_inv) > 0), np.diag(s2_inv)
        assert np.all(np.diag(s2_p_inv) > 0), np.diag(s2_p_inv)
        assert np.all(np.diag(s2_0_inv) > 0)

        # Diagonal entries
        AA = np.zeros((t_pts, lat_dim, lat_dim))
        # Upper diagonal entries
        BB = np.zeros((t_pts-1, lat_dim, lat_dim))

        s2t0 = s2_0_inv
        s2t1 = s2_inv if v[1] == 0 else s2_p_inv
        for t in range(t_pts-1):
            dt = 1.0 if t == 0 else times[t] - times[t-1]
            dt1 = times[t+1] - times[t]

            AA[t] = (s2t0/dt + s2t1/dt1)
            BB[t] = -s2t1/dt1

            assert np.all(np.diag(AA[t]) > 0)

            s2t0 = np.copy(s2t1)
            s2t1 = s2_inv if v[t+1] == 0 else s2_p_inv

        # last time point
        dt = times[t_pts-1] - times[t_pts-2]
        if v[t_pts-1] == 0:
            AA[t_pts-1] = s2_inv/dt
        else:
            AA[t_pts-1] = s2_p_inv/dt
        return AA, BB
