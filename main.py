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

def estimate_alr(Y):
    # alr
    X = []
    ntaxa = Y[0].shape[1]
    prior = np.ones(ntaxa)
    X_dm = []
    for y in Y:
        p = (y + prior) / (y + prior).sum(axis=1, keepdims=True)
        X_dm.append( (np.log(p[:,:-1]).T - np.log(p[:,-1])).T )
    return X_dm


def train(Y, U, T, input_dir, output_dir):
    if input_dir is not None:
        if not os.path.exists(input_dir):
            print("Directory", input_dir, "does not exists", file=sys.stderr)
        try:
            X = pkl.load(open(input_dir + "/X.pkl", "rb"))
        except FileNotFoundError:
            print("Estimating relative abundances...", file=sys.stderr)
            vmlds = NoisyVMLDS(Y, U, T)
            vmlds.optimize(verbose=True)
            X = vmlds.get_latent_means()
            pkl.dump(X, open(output_dir + "/X.pkl", "wb"))
    else:
        print("Estimating relative abundances...", file=sys.stderr)
        vmlds = NoisyVMLDS(Y, U, T)
        vmlds.optimize(verbose=True)
        X = vmlds.get_latent_means()
        pkl.dump(X, open(output_dir + "/X.pkl", "wb"))
    # X = estimate_alr(Y)
    print("Running parameter estimation...", file=sys.stderr)
    clv = CompositionalLotkaVolterra(X, Y, T, U)
    clv.train(verbose=True)
    A, g, B = clv.get_params()
    print("Saving parameters to", output_dir)
    np.savetxt(output_dir + "/A", A)
    np.savetxt(output_dir + "/g", g)
    np.savetxt(output_dir + "/B", B)


def predict(Y, U, T, IDs, A, g, B, otu_table, output_dir):
    print("Computing predictions...")
    clv = CompositionalLotkaVolterra()
    clv.A = A
    clv.g = g
    clv.B = B

    Y_pred = []
    for y,u,t in zip(Y, U, T):
        y0 = y[0]
        y_pred = clv.predict(y0,t,u)
        Y_pred.append(y_pred)

    util.write_table(IDs, Y_pred, T, otu_table, output_dir, postfix="pred")


def estimate(Y, U, T, IDs, otu_table, output_dir):
    print("Estimating relative abundances...", file=sys.stderr)

    model = NoisyVMLDS(Y, U, T)
    model.optimize(verbose=True)
    X = model.get_latent_means()
    var = model.gamma2
    pkl.dump(X, open(output_dir + "/X.pkl", "wb"))
    Y_pred = []
    W_pred = []
    for x in X:
        tpts = x.shape[0]
        zeros = np.zeros((tpts, 1))

        x = np.hstack( (x + var, zeros) )
        p = np.exp(x - logsumexp(x, axis=1,keepdims=True))
        Y_pred.append(p)

    util.write_table(IDs, Y_pred, T, otu_table, output_dir, postfix="est")

    W = model.W
    util.write_table(IDs, W, T, otu_table, output_dir, postfix="nonzero-posterior-probs")


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

    args = parser.parse_args(argv[1:])
    cmd = args.command
    otu_table = args.otu_table
    event_table = args.events
    input_dir = args.indir.strip("/") if args.indir is not None else None
    output_dir = args.outdir.strip("/") if args.outdir is not None else None

    IDs, Y, U, T = util.load_observations(otu_table, event_table)

    if cmd == "train":
        train(Y, U, T, input_dir, output_dir)
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
            print("Please specify directory to load model parameters.")
            print("Did you run")
            print("\tpython main.py train", otu_table)
            print("first?")
        predict(Y, U, T, IDs, A, g, B, otu_table, output_dir)
    elif cmd == "estimate":
        estimate(Y, U, T, IDs, otu_table, output_dir)

    else:
        print("unrecognized command", file=sys.stdout)