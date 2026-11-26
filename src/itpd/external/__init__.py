"""External baselines: PCMCI, DF-PC, DYNOTEARS and CD-NOD.

Each baseline is one script in this folder (`pcmci.py`, `dfpc.py`,
`dynotears.py`, `cdnod.py`). A script runs alone in the Python interpreter of
its tool and does not import `itpd`. The folder `requirements/` holds the
package versions of each interpreter. `run_external` writes the data to a file,
starts the script and reads its result. Environment variables name the
interpreters:

    ITPD_PYTHON_PCMCI, ITPD_PYTHON_DFPC, ITPD_PYTHON_DYNOTEARS, ITPD_PYTHON_CDNOD
    ITPD_DFPC_SOURCE    folder of the DF-PC clone

When a variable is not set, the run has the status "skipped".
"""
import json
import os
import subprocess
import tempfile

import numpy as np

SCRIPTS = {"PCMCI": "pcmci.py", "DF-PC": "dfpc.py", "DYNOTEARS": "dynotears.py", "CD-NOD": "cdnod.py"}
INTERPRETERS = {"PCMCI": "ITPD_PYTHON_PCMCI", "DF-PC": "ITPD_PYTHON_DFPC",
                "DYNOTEARS": "ITPD_PYTHON_DYNOTEARS", "CD-NOD": "ITPD_PYTHON_CDNOD"}
ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def corrected_level(name, alpha, N, T, max_lag=1):
    """Return `alpha` divided by the number of candidate parents.

    - PCMCI and DF-PC: N (T - 1) - 1.
    - CD-NOD: N * max_lag."""
    if name == "CD-NOD":
        return alpha / (N * max_lag)
    return alpha / (N * (T - 1) - 1)


def _skipped(name, reason):
    print(f"{name} skipped: {reason}")
    return {"status": "skipped", "message": reason}


def run_external(name, X, *, time_limit=None, **options):
    """Run one external baseline on data X (M, T, N) and return its result.

    `options` are the options of the script:
    - PCMCI, CD-NOD: `alpha` and `max_lag`
    - DF-PC: `alpha`
    - DYNOTEARS: `max_lag` and `unit_variance`

    `time_limit` is in seconds. DF-PC checks it between two CI tests; the other
    scripts are stopped from outside.

    The result has status "ok", "timeout", "skipped" or "error".
    - With "ok", it holds `A_hat` (the time graph, T N x T N, Boolean),
      `seconds`, `versions` and, for PCMCI and DF-PC, `requested`, `unique`,
      `largest_set` and `mean_set`.
    - With "skipped" or "error", it holds a message."""
    python = os.environ.get(INTERPRETERS[name])
    if not python:
        return _skipped(name, f"{INTERPRETERS[name]} is not set")
    if not os.path.isfile(python):
        return _skipped(name, f"{INTERPRETERS[name]} names no file")
    if name == "DF-PC":
        source = os.environ.get("ITPD_DFPC_SOURCE")
        if not source or not os.path.isdir(os.path.join(source, "cddd")):
            return _skipped(name, "ITPD_DFPC_SOURCE does not name a DF-PC clone")
        options = dict(options, source=source, time_limit=time_limit)
        time_limit = None
    # One BLAS thread: the numbers must not depend on the number of cores.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update({k: "1" for k in ONE_THREAD}, PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory() as folder:
        data, result = os.path.join(folder, "in.npz"), os.path.join(folder, "out.npz")
        np.savez(data, X=np.asarray(X, dtype=np.float64))
        command = [python, "-B", os.path.join(os.path.dirname(__file__), SCRIPTS[name]),
                   "--input", data, "--output", result]
        for key, value in options.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, (bool, np.bool_)):
                command += [flag] if value else []
            elif value is not None:
                command += [flag, repr(float(value)) if isinstance(value, (float, np.floating)) else str(value)]
        try:
            done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=time_limit)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if done.returncode != 0:
            return {"status": "error", "message": done.stderr[-2000:]}
        with np.load(result, allow_pickle=False) as z:
            out = {key: z[key] if key == "A_hat" else z[key].item() for key in z.files}
    out["versions"] = json.loads(out["versions"])
    return out
