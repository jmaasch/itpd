"""PCMCI (tigramite) on data of shape (M, T, N).

    python pcmci.py --input in.npz --output out.npz --alpha 0.01 --max-lag 1

One PCMCI run pools the M observations and all time steps
(analysis_mode="multiple"). It returns one window graph with lags
1, ..., max-lag, which is repeated at every time step. A link is in the graph
when its p-value is at most alpha; alpha is also the level of the condition
selection. The lag-1 self link of every series is given as known: PCMCI
does not test it and always conditions on it. The p-value of a partial
correlation is the Fisher-z p-value. Every call of the test is counted, with
the variables in lag coordinates.
"""
import time

import numpy as np
from scipy.stats import norm
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

import _common

# tigramite drops the first 2 * max-lag steps of every observation.
CUT_OFF = "2xtau_max"


class FisherZParCorr(ParCorr):
    """Partial correlation with the Fisher-z p-value and a test counter."""

    def __init__(self, N, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.counter = _common.TestCounter()

    def get_analytic_significance(self, value, T, dim, xyz):
        df = T - dim - 1
        if df <= 0:
            raise ValueError("The Fisher-z test needs more rows than the conditioning set has members plus 3")
        r = min(max(float(value), -0.9999999), 0.9999999)
        return float(2 * (1 - norm.cdf(abs(np.sqrt(df) * np.arctanh(r)))))

    def _code(self, variable):
        """Return the lag coordinate of a (series, lag) pair with lag <= 0."""
        return -variable[1] * self.N + variable[0]

    def run_test(self, X, Y, Z=None, tau_max=0, cut_off=None, alpha_or_thres=None):
        self.counter.record(self._code(X[0]), self._code(Y[0]), [self._code(z) for z in Z or []])
        return super().run_test(X, Y, Z=Z, tau_max=tau_max, cut_off=cut_off or CUT_OFF,
                                alpha_or_thres=alpha_or_thres)


def main():
    p = _common.parser(__doc__.split("\n")[0])
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--max-lag", type=int, default=1)
    args = p.parse_args()
    X = _common.read_input(args.input)
    M, T, N = X.shape
    K, alpha = args.max_lag, args.alpha
    if T <= 2 * K:
        raise ValueError(f"No rows are left: T = {T} and 2 * max-lag = {2 * K}")
    frame = pp.DataFrame({m: X[m] for m in range(M)}, analysis_mode="multiple")
    pcmci = PCMCI(dataframe=frame, cond_ind_test=FisherZParCorr(N, significance="analytic"), verbosity=0)
    links = {j: {(i, -l): "-->" if (i == j and l == 1) else "-?>"
                 for i in range(N) for l in range(1, K + 1)} for j in range(N)}
    start = time.perf_counter()
    result = pcmci.run_pcmci(tau_min=1, tau_max=K, pc_alpha=alpha, alpha_level=alpha, fdr_method="none",
                             link_assumptions=links)
    seconds = time.perf_counter() - start
    win = np.zeros((K + 1, N, N), dtype=bool)
    for l in range(1, K + 1):
        win[l] = result["p_matrix"][:, :, l] <= alpha        # [i, j, l]: series i at lag l -> series j
    # PCMCI holds a copy of the test, so the counter is read from that copy.
    _common.write_output(args.output, _common.window_to_time_graph(win, T), seconds,
                         ["tigramite", "numpy", "scipy"], counter=pcmci.cond_ind_test.counter)


if __name__ == "__main__":
    main()
