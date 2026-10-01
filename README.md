# Iterative Temporal Parent Discovery

Iterative Temporal Parent Discovery (ITPD) is a constraint-based causal discovery algorithm for multivariate time series.

## Repository structure

```
.
├── itpd.py               # ITPD with CI test reduction.
├── padl_itpd.py          # The version of PaDL used by ITPD.
├── itpd_naive.py.        # ITPD_naive (no CI test reduction).
├── padl_naive.py.        # The version of PaDL used by ITPD_naive.
├── itpd_test.ipynb       # A notebook comparing ITPD to ITPD_naive with an oracle independence test.
├── itpd_naive_test.ipynb # A notebook using ITPD_naive with finite data simulation (finite data have errors; just for ITPD syntax demo).
├── LICENSE
└── README.md
```
