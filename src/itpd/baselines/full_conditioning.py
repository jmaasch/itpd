"""FullCI (`full_conditioning`): one CI test per candidate given all other earlier nodes."""
from ._targets import Result, run_per_target


def _parents(pvalue, x, y, candidates, alpha):
    """`find_parents` function of FullCI for `run_per_target`.

    Return candidates z with pvalue(y, z, S) <= alpha, where S is x and all
    other `candidates`.
    """
    earlier = sorted([x, *candidates])
    # The first test that cannot be evaluated ends the target.
    return [z for z in candidates
            if pvalue(y, z, [s for s in earlier if s != z]) <= alpha]


def full_conditioning(counter, N: int, T: int, alpha: float = 0.01, *,
                      corrected: bool = False) -> Result:
    """Run FullCI on N series and T steps.

    A candidate z is a parent of the target y when z and y are dependent given
    all other nodes of the earlier steps. `counter` is a `CICounter`. See
    `run_per_target` for `alpha` and `corrected`.
    """
    return run_per_target(_parents, counter, N, T, alpha, corrected)
