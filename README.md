# Log-Likelihood Simulation
Code to simulate from and characterize the distribution of the log-likelihood ratio for the noise model.

The following experiments are performed.

| File Name | Num Grid | Grid LB | Grid UB | h           | Number of Samples | q    | Bisecting Mode | Analytical Solver | Number of CPU | Notes |
| --------- | -------- | ------- | ------- | ------------| ----------------- | ---- | -------------- | ----------------- | ------------- | ----- |
| exp1.npz  | 20       | 0       | 3       | (0.5 0.5)   | 100k              | 0.67 | False          | True              | 12            |       |
| exp2.npz  | 20       | 0       | 3       | (0.25 0.75) | 100k              | 0.67 | False          | False             | 12            |       |
| exp3.npz  | 20       | 0       | 3       | (1 -1)      | 100k              | 0.67 | False          | True              | 12            | INVALID...only valid for true mu=0.       |
| exp4.npz  | 20       | 0       | 3       | (1 -1)      | 10k               | 0.67 | True           | False             | 8             |       |
| exp5.npz  | 30       | 0       | 10      | (0.5 0.5)   | 10k               | 0.67 | False          | False             | 12            |       |
| exp6.npz  | 20       | 0       | 0.25    | (0.5 0.5)   | 100k              | 0.67 | False          | True              | 12            | Saved as exp1_point25.npz      |
| exp7.npz  | 20       | 0       | 3      | (1 -1)   | 50k               | 0.67 | False          | True              | 12            | None  |