import numpy as np

def write_table(IDs, observations, time_points, otus, outfile):
    ntaxa = observations[0].shape[1]
    otu_table = []
    for idx, obs in enumerate(observations):
        for t,obs_t in enumerate(obs):
            header = np.array([IDs[idx], time_points[idx][t]])
            row = np.concatenate((header, obs_t))
            otu_table.append(row)
    otu_table = np.array(otu_table).T
    col1 = np.concatenate( (np.array(["id", "time"]), otus) )
    col1 = np.expand_dims(col1, axis=1)
    otu_table = np.hstack((col1, otu_table.astype(str)))

    if outfile[-4:] == ".csv":
        np.savetxt(outfile, otu_table, delimiter=",", fmt="%s")
    else:
        np.savetxt(outfile + ".csv", otu_table, delimiter=",", fmt="%s")