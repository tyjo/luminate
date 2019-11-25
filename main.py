import numpy as np
import os
import pickle as pkl
import sys
from argparse import ArgumentParser
from scipy.misc import logsumexp
from sys import argv

import src.util as util
from src.compositional_lotka_volterra import CompositionalLotkaVolterra
from src.noisy_vmlds import NoisyVMLDS
from src.find_stable_subset import find_stable_subset


def choose_denom(Y, otu_table):
    """Find a suitable denominator for the additive log
    ratio transformation (used for NoisyvMLDS).
    """
    ntaxa = Y[0].shape[1]
    avg_nonzero = np.zeros(ntaxa)
    avg_abun = np.zeros(ntaxa)
    avg_var = np.zeros(ntaxa)
    for y in Y:
        avg_y = np.sum(y!=0,axis=0) / y.shape[0]
        avg_nonzero += avg_y

        y_norm = y / y.sum(axis=1,keepdims=True)
        for d in range(ntaxa):
            avg_var[d] += np.var(y_norm[:,d])
        avg_abun += y_norm.sum(axis=0)/y.shape[0]
    avg_nonzero /= len(Y)
    avg_abun /= len(Y)
    avg_var /= len(Y)

    otu_table = np.loadtxt(otu_table, delimiter=",", dtype=str)
    taxon_names = otu_table[2:,0]

    low_var_idxes = np.argsort(avg_var)
    low_var_idx = None
    for idx in low_var_idxes:
        if avg_nonzero[idx] > 0.8:
            low_var_idx = idx
            break

    taxon = taxon_names[low_var_idx]
    row = low_var_idx
    avg = avg_nonzero[low_var_idx]
    return row, taxon


def train(Y, U, T, event_names, denom, input_dir, output_dir, otu_table, bootstrap_replicates):
    """Train cLV.
    """
    if input_dir is not None:
        if not os.path.exists(input_dir):
            print("Directory", input_dir, "does not exists", file=sys.stderr)
        try:
            P = pkl.load(open(input_dir + "/P.pkl", "rb"))
        except FileNotFoundError:
            print("Estimating relative abundances...", file=sys.stderr)
            vmlds = NoisyVMLDS(Y, U, T, denom)
            vmlds.optimize(verbose=True)
            P = vmlds.get_relative_abundances()
            pkl.dump(P, open(output_dir + "/P.pkl", "wb"))


    else:
        print("Estimating relative abundances...", file=sys.stderr)
        vmlds = NoisyVMLDS(Y, U, T, denom)
        vmlds.optimize(verbose=True)
        P = vmlds.get_relative_abundances()
        pkl.dump(P, open(output_dir + "/P.pkl", "wb"))


    print("Running parameter estimation...")
    clv = CompositionalLotkaVolterra(P, T, U)
    clv.train(verbose=True)
    A, g, B = clv.get_params()
    print("Saving parameters to", output_dir)
    np.savetxt(output_dir + "/A", A)
    np.savetxt(output_dir + "/g", g)
    np.savetxt(output_dir + "/B", B)

    otu_table = np.loadtxt(otu_table, delimiter=",", dtype=str)
    taxon_names = otu_table[2:,0].tolist()
    np.savetxt(output_dir + "/dimensions", taxon_names, fmt="%s")
    np.savetxt(output_dir + "/effect-dimensions", event_names, fmt="%s")

    # bootstrap resampling
    if bootstrap_replicates > 0:
        print("Performing bootstrap estimation...")
        print("\tMinimum one-sided p-value:", 1./bootstrap_replicates)

        nsamples = len(Y)
        if nsamples < 30:
            print("\tWarning: sample size may be too small for bootstrap resampling.", file=sys.stderr)

        A_prob = np.zeros(A.shape)
        g_prob = np.zeros(g.shape)
        B_prob = np.zeros(B.shape)

        for j in range(bootstrap_replicates):
            print("\tPerforming bootstrap replicate {} of {}".format(j+1, bootstrap_replicates))
            P_bs = []
            U_bs = []
            T_bs = []

            while len(P_bs) < nsamples:
                idx = np.random.randint(nsamples)
                P_bs.append(np.copy(P[idx]))
                U_bs.append(np.copy(U[idx]))
                T_bs.append(np.copy(T[idx]))

            clv_bs = CompositionalLotkaVolterra(P_bs, T_bs, U_bs)
            alpha, r_A, r_g, r_B = clv.get_regularizers()
            clv_bs.set_regularizers(alpha, r_A, r_g, r_B)
            clv_bs.train(verbose=False)
            A_bs, g_bs, B_bs = clv_bs.get_params()

            A_prob += np.logical_and(A > 0, A_bs > 0).astype(float)
            A_prob += np.logical_and(A < 0, A_bs < 0).astype(float)

            B_prob += np.logical_and(B > 0, B_bs > 0).astype(float)
            B_prob += np.logical_and(B < 0, B_bs < 0).astype(float)

            g_prob += np.logical_and(g > 0, g_bs > 0).astype(float)
            g_prob += np.logical_and(g < 0, g_bs < 0).astype(float)

        A_prob /= bootstrap_replicates
        #A_prob[A_prob == 1] = 1-1./bootstrap_replicates
        g_prob /= bootstrap_replicates
        #g_prob[g_prob == 1] = 1-1./bootstrap_replicates
        B_prob /= bootstrap_replicates
        #B_prob[B_prob == 1] = 1-1./bootstrap_replicates

        np.savetxt(output_dir + "/A_pval", 1 - A_prob, fmt="%.4f")
        np.savetxt(output_dir + "/g_pval", 1 - g_prob, fmt="%.4f")
        np.savetxt(output_dir + "/B_pval", 1 - B_prob, fmt="%.4f")


def predict(Y, U, T, IDs, A, g, B, otu_table, output_dir, one_step=False):
    """
    """
    print("Computing predictions...")

    # need this to compute the same denominator as above
    P = pkl.load(open(input_dir + "/P.pkl", "rb"))
    clv = CompositionalLotkaVolterra(P, U, T)
    clv.A = A
    clv.g = g
    clv.B = B
    
    P_pred = []
    for y,u,t in zip(Y, U, T):
        p = y / y.sum(axis=1,keepdims=True)
        p = (p + 1e-3) / (p + 1e-3).sum(axis=1,keepdims=True)

        if one_step:
            p_pred = clv.predict_one_step(p,t,u)
        else:
            p_pred = clv.predict(p,t,u)

        P_pred.append(p_pred)

    util.write_table(IDs, P_pred, T, otu_table, output_dir, postfix="pred")


def estimate(Y, U, T, IDs, denom, otu_table, output_dir):
    """Estimate relative abundances a biological zero
    posterior probabilities.
    """
    print("Estimating relative abundances...")

    np.set_printoptions(suppress=True)
    model = NoisyVMLDS(Y, U, T, denom)
    model.optimize(verbose=True)
    P_pred = model.get_relative_abundances()
    pkl.dump(P_pred, open(output_dir + "/P.pkl", "wb"))

    util.write_table(IDs, P_pred, T, otu_table, output_dir, postfix="est")

    W = model.get_posterior_nonzero_probs()
    util.write_table(IDs, W, T, otu_table, output_dir, postfix="nonzero-posterior-probs")


def plot_trajectories(IDs, Y, U, T, effect_names, taxon_names, output_dir, outfile):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def plot_bar(ax, y, time, unique_color_id, remaining_ids):
        T = y.shape[0]
        cm = plt.get_cmap("tab20c")
        colors = [cm(i) for i in range(20)]
        #time = np.array([t for t in range(T)])
        widths = np.concatenate((time[1:] - time[:-1], [1]))
        widths[widths > 1] = 1
        widths -= 1e-1

        y_colors = y[:,unique_color_id]
        names = taxon_names[unique_color_id]
        ax.bar(time, y_colors[:,0], width=widths, color=colors[0], align="edge", label=names[0])
        for j in range(1, y_colors.shape[1]):
            ax.bar(time, y_colors[:,j], bottom=y_colors[:,:j].sum(axis=1), width=widths, color=colors[j], align="edge", label=names[j])
        ax.bar(time, y[:,remaining_ids].sum(axis=1), bottom=y_colors.sum(axis=1), width=widths, color=colors[19], align="edge", label="Aggregate")
        #ax.set_title("Relative Abundances", fontsize=10)
        ax.legend(prop={"size" : 4}, bbox_to_anchor=[-0.1,1.225], loc="upper left", ncol=4)

    def plot_effects(ax, u, time, yticklabels, title):
        T = u.shape[0]
        cm = plt.get_cmap("tab20c")
        colors = [cm(i) for i in range(20)]
        widths = np.concatenate((time[1:] - time[:-1], [1]))
        widths[widths > 1] = 1
        widths -= 1e-1
        u = np.copy(u)
        u[u > 0] = 1

        ax.bar(time, u[:,0], width=widths, color=colors[0], align="edge")
        for j in range(1, u.shape[1]):
            ax.bar(time, u[:,j], bottom=j, width=widths, color=colors[j%20], align="edge")
        ax.set_title(title, fontsize=10)
        ax.set_yticks([i for i in range(u.shape[1]+1)])
        ax.set_yticklabels(np.concatenate([[""], yticklabels]))
        ax.set_xticklabels([])

    def find_top_ids(Y, n):
        ntaxa = Y[0].shape[1]
        rel_abun = np.zeros(ntaxa)
        for y in Y:
            tpts = y.shape[0]
            denom = y.sum(axis=1,keepdims=True)
            denom[denom == 0] = 1
            p = y / denom
            rel_abun += p.sum(axis=0) / tpts
        ids = np.argsort(-rel_abun)
        return np.sort(ids[:n]), np.sort(ids[n:])

    N = len(Y)
    top19_ids, remaining_ids = find_top_ids(Y, 19)
    n_effects = U[0].shape[1]
    for i in range(N):
        fig = plt.figure(figsize=(4,4))
        gs = gridspec.GridSpec(2, 1, height_ratios=[n_effects,6*n_effects])
        gs.update(hspace=1)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        denom = Y[i].sum(axis=1)
        denom[denom == 0] = 1

        plot_effects(ax1, U[i], T[i], effect_names, "Effects")
        plot_bar(ax2, (Y[i].T / denom).T, T[i], top19_ids, remaining_ids)


        outfile = os.path.splitext(outfile)[0]
        gs.tight_layout(fig, h_pad=3.5)

        plt.savefig(output_dir + "/" + outfile + "-{}.pdf".format(IDs[i]))
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Time-series modeling for the microbiome")
    parser.add_argument("command", type=str,
                                   help="Specify analysis to run. One of: " + \
                                        "train,predict,estimate.")
    parser.add_argument("otu_table", type=str, metavar="otu-table",
                                     help="Filepath to OTU table csv.")
    parser.add_argument("-e", "--events", type=str, default="",
                                          help="Filepath to table of external events.")
    parser.add_argument("-o", "--outdir", type=str, default="",
                                          help="Specify output directory to store results. " + \
                                               "Default is current directory.")
    parser.add_argument("-i", "--indir", type=str, default=None,
                                         help="Specify input directory to load previously " + \
                                         "computed parameters: typically the OUTDIR from a " + \
                                         "previous run.")
    parser.add_argument("-b", "--bootstrap", type=int, default=0,
                                             help="Perform bootstrap resampling to estimate one-sided " + \
                                             "p-values of cLV coefficients. The argument specifies the " + \
                                             "number of bootstrap replicates to perform. Will produce a warning " + \
                                             "if the sample size is too small (N<30).")
    parser.add_argument("-s", "--one-step", default=False, action="store_true",
                                             help="Perform one-step prediction instead of prediction " + \
                                                  "from initial conditions.")

    args = parser.parse_args(argv[1:])
    cmd = args.command
    otu_table = args.otu_table
    event_table = args.events
    input_dir = args.indir.strip("/") if args.indir is not None else None
    output_dir = args.outdir.strip("/") if args.outdir is not None else None
    bootstrap_replicates = args.bootstrap
    one_step = args.one_step

    IDs, Y, U, T, event_names = util.load_observations(otu_table, event_table)

    if cmd == "train":    
        # find an appropriate denominator for the denoising step
        denom, taxon = choose_denom(Y, otu_table)
        train(Y, U, T, event_names, denom, input_dir, output_dir, otu_table, bootstrap_replicates)
    elif cmd == "predict":
        if input_dir is not None:
            try:
                print("Loading model parameters from", input_dir, file=sys.stderr)
                A = np.loadtxt(input_dir + "/A")
                g = np.loadtxt(input_dir + "/g")
                B = np.loadtxt(input_dir + "/B")
                if B.ndim == 1:
                    B = np.expand_dims(B,axis=1)
            except OSError:
                print("Unable to load parameters")
        else:
            print("Please specify directory to load model parameters.", file=sys.stderr)
            print("Did you run", file=sys.stderr)
            print("\tpython main.py train", otu_table, file=sys.stderr)
            print("first?", file=sys.stderr)
        predict(Y, U, T, IDs, A, g, B, otu_table, output_dir, one_step)
    elif cmd == "estimate":
        # find an appropriate denominator for the denoising step
        denom, taxon = choose_denom(Y, otu_table)
        estimate(Y, U, T, IDs, denom, otu_table, output_dir)

    elif cmd == "plot":
        taxon_names = np.loadtxt(otu_table, dtype=str, delimiter=",")[2:,0]
        plot_trajectories(IDs, Y, U, T, event_names, taxon_names, output_dir, os.path.basename(otu_table))

    else:
        print("unrecognized command", file=sys.stderr)