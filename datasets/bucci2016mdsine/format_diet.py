import numpy as np
import pickle as pkl

from util_bucci import write_table
sample_id_to_subject_id = {}
subject_id_time = {}


with open("data_diet/metadata.txt", "r") as f:
    for line in f:
        line = line.split()

        if "sampleID" in line[0]:
            continue

        sample_id = line[0]
        subject_id = line[2]
        day = float(line[3])

        sample_id_to_subject_id[sample_id] = subject_id
        subject_id_time[subject_id] = subject_id_time.get(subject_id, []) + [day]


counts = np.loadtxt("data_diet/counts.txt", delimiter="\t", dtype=str, comments="!")
otus = counts[1:,0]
# swap last two rows since there are no zeros in the penultimate row
# tmp = counts[-2]
# counts[-2] = counts[-1]
# counts[-1] = tmp
counts = counts[:,1:]

subject_id_counts = {}

for row in counts.T:
    sample_id = row[0]
    counts = row[1:].astype(float)
    subject_id = sample_id_to_subject_id[sample_id]

    counts /= 1000
    if subject_id in subject_id_counts:
        subject_id_counts[subject_id] = np.vstack( (subject_id_counts[subject_id], np.array(counts)) )
    else:
        subject_id_counts[subject_id] = np.array(counts)


e = open("diet-events.csv", "w")
e.write("ID,eventID,startDay,endDay\n")
Y_diet = []
T_diet = []
IDs_diet = []
zero_counts = 0
total_counts = 0
for subject_id in subject_id_counts:
    y = np.array(subject_id_counts[subject_id])
    t = np.array(subject_id_time[subject_id])
    zero_counts += y[y == 0].size
    total_counts += y.size

    Y_diet.append(y)
    T_diet.append(t)
    IDs_diet.append(subject_id)

    # long observations have change in diet
    if y.shape[0] > 29:
        e.write("{},Low fiber diet,35,49\n".format(subject_id))

#print("% zero", zero_counts / total_counts)
write_table(IDs_diet, Y_diet, T_diet, otus, "diet")