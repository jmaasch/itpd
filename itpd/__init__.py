"""itpd: instrumented ITPD, ITPD_naive and an order-based baseline on one CI-test foundation.

Conventions used everywhere
- Time is 0-based; the process starts at t = 0 (those nodes are roots). The paper's V_1 is t = 0 here.
- A node (variable n, time t) has the global column index col = t * N + n (time-major).
- A graph matrix A has A[i, j] = 1 iff i -> j.
"""
