# LUMINATE
LUMINATE (longitudinal microbiome inference and zero detection) includes three programs for inference in microbiome datasets:

* `estimate`: estimate relative abundances and posterior probabilities of biological zeros.
* `train`: estimate model parameters for compositional Lotka-Volterra (cLV).
* `predict`: predict longitudinal trajectories from initial conditions.

## Quick Start
See `./run-examples.sh` to run each program on an example dataset.


## Input

Each takes as input an OTU Table (taxa by samples) with the following structure:

*  The first row gives IDs for each longitudinal sequence.
*  The second row gives sample times.
*  The first column gives taxon names.

See `datasets/bucci2016mdsine/cdiff-counts.csv` for an example.

A second optional table providing external perturbations can be specified. This is a .csv file where the columns are `sequenceID,eventID,startDay,endDay,magnitude(optional)`

The first column links an event to a longitudinal sample through `sequenceID`. This should correspond to an ID in the OTU Table. The second column gives a name for each event: parameters are inferred for each unique event name.

The start day and stop day give a range of time over which an event occurs (end point included). For example, these could be a range of time when a patient receives antibiotics. For one time events, set `startDay=stopDay.`

The `magnitude` column is used to estimate the parameters of cLV. If unspecified, events are treated like indicator variables: set to 1 during the time of an event and 0 otherwise.

## Complete Example
Here we run through a typical workflow for LUMINATE using the data in `bucci2016mdsine`. First, we want to estimate relative abundances and save the results:

```
python main.py estimate
               datasets/bucci2016mdsine/cdiff-counts.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -o datasets/bucci2016mdsine
```

The first positional argument specifies the type of analysis. The second position argument specifies the path to the OTU table. The optional argument `-e` (`--effects`) specifies a filepath to the .csv of external events. Finally `-o` (`--outdir`) gives a directory to store results for downstream analysis.

Running `estimate` produces 3 files:

* `cdiff-counts-est.csv` : the estimated relative abundances
* `cdiff-counts-nonzero-posterior-prob.csv`: posterior probabilities of sampling zeros (i.e. a taxon is below the detection threshold as opposed to absent from the community)
* `X.pkl` : a temporary result used by downstream programs.


Next, we want to estimate the parameters of cLV using these estimates:

```
python main.py train \
               datasets/bucci2016mdsine/cdiff-counts.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -i datasets/bucci2016mdsine \
               -o datasets/bucci2016mdsine
```

The `-i` (`--indir`) flag tells LUMINATE to look in this directory for estimated relative abundances (`X.pkl`). If this is not found, then `estimate` will run by default. The `-o` (`--outdir`) flag tells LUMINATE to save model parameters to this folder.

Finally, if we want to predict trajectories from initial conditions, we can call

```
python main.py predict \
               datasets/bucci2016mdsine/cdiff-counts.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -i datasets/bucci2016mdsine \
               -o datasets/bucci2016mdsine
```

This produces an OTU table of predicted trajectories from initial conditions. The first observation from `cdiff-counts.csv` are used as initial conditions to forecast the remaining observations in the sequence.

A new OTU table `cdiff-counts-pred.csv` is produced with the same IDs and sample times as `cdiff-counts.csv` with predicted relative abundances. Note that the first observation in each sequence is set to `0` since these were used as initial conditions.

## Program Options

```
python main.py -h
usage: main.py [-h] [-e EVENTS] [-o OUTDIR] [-i INDIR] command otu-table

Time-series modeling for the microbiome

positional arguments:
  command               Specify analysis to run. One of:
                        train,predict,estimate.
  otu-table             Filepath to OTU table csv.

optional arguments:
  -h, --help            show this help message and exit
  -e EVENTS, --events EVENTS
                        Filepath to table of external events.
  -o OUTDIR, --outdir OUTDIR
                        Specify output directory to store results. Default is
                        current directory.
  -i INDIR, --indir INDIR
                        Specify input directory to load previously computed
                        parameters: typically the OUTDIR from a previous run.
```
