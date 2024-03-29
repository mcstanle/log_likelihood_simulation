# Log-Likelihood Simulation
Code to simulate from and characterize the distribution of the log-likelihood ratio for the noise model.

The following experiments are performed.

| File Name | Num Grid | Grid LB  | Grid UB  | h           | Number of Samples | q    | Bisecting Mode | Analytical Solver | Number of CPU | Notes                               |
| --------- | -------- | -------  | -------  | ------------| ----------------- | ---- | -------------- | ----------------- | ------------- | ----------------------------------- |
| exp1.npz  | 20       | 0        | 3        | (0.5 0.5)   | 100k              | 0.67 | False          | True              | 12            |                                     |
| exp2.npz  | 20       | 0        | 3        | (0.25 0.75) | 100k              | 0.67 | False          | False             | 12            |                                     |
| exp3.npz  | 20       | 0        | 3        | (1 -1)      | 100k              | 0.67 | False          | True              | 12            | INVALID...only valid for true mu=0. |
| exp4.npz  | 20       | 0        | 3        | (1 -1)      | 10k               | 0.67 | True           | False             | 8             |                                     |
| exp5.npz  | 30       | 0        | 10       | (0.5 0.5)   | 10k               | 0.67 | False          | False             | 12            |                                     |
| exp6.npz  | 20       | 0        | 0.25     | (0.5 0.5)   | 100k              | 0.67 | False          | True              | 12            | Saved as exp1_point25.npz           |
| exp7.npz  | 20       | 0        | 3        | (1 -1)      | 50k               | 0.67 | False          | True              | 12            | None                                |
| exp8.npz  | 07       | 0        | 1        | (1 -1)      | 50k               | 0.95 | False          | False             | 12            | None                                |
| exp9.npz  | (40x20)  | (0, 0)   | (4, 2)   | (1 -1)      | 50k               | 0.95 | False          | True              | 12            | None                                |
| exp10.npz | (40x20)  | (0, 0)   | (4, 2)   | (1 -1)      | 50k               | 0.5  | False          | True              | 12            | None                                |
| exp11.npz | (40x20)  | (0, 0)   | (4, 2)   | (1 -1)      | 50k               | 0.68 | False          | True              | 12            | None                                |
| exp12.npz | (40x20)  | (0, 0)   | (3, 5)   | (1 -1)      | 50k               | 0.68 | False          | True              | 12            | None                                |
| exp13.npz | (40x40)  | (-3, 0)  | (-3, 0)  | (1 -1)      | 50k               | 0.95 | False          | True              | 12            | log_e for grid                      |