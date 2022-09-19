# Log-Likelihood Simulation
Code to simulate from and characterize the distribution of the log-likelihood ratio for the noise model.

The following experiments are performed.

| File Name | Num Grid | Grid LB | Grid UB | h           | Number of Samples | q    | Bisecting Mode | Analytical Solver | Number of CPU |
| --------- | -------- | ------- | ------- | ------------| ----------------- | ---- | -------------- | ----------------- | ------------- |
| exp1.npz  | 20       | 0       | 3       | (0.5 0.5)   | 100k              | 0.67 | False          | True             | 12             |
| exp2.npz  | 20       | 0       | 3       | (0.25 0.75) | 10k               | 0.67 | False          | False             | 8             |
| exp3.npz  | 20       | 0       | 3       | (1 -1)      | 10k               | 0.67 | False          | False             | 8             |
| exp4.npz  | 20       | 0       | 3       | (1 -1)      | 10k               | 0.67 | True           | False             | 8             |
| exp5.npz  | 30       | 0       | 10      | (0.5 0.5)   | 10k               | 0.67 | False          | False             | 12            |