# Lean 4 proof that ITPD returns the parent set

This folder holds a machine-checked proof for Algorithm 1 (ITPD). It proves that Algorithm 1 returns `A[Y] = pa(Y)` for every target `Y` whose parents are all observed, when the CI answers match the separation statements of the graph.

- Lean 4.33.1, core library only. No Mathlib, no other dependency.
- 7 modules, 61 theorem declarations (supporting lemmas included). Every proof is complete: no `sorry`. The library declares no axiom.
- The graph: every edge points strictly forward in time. Every series has its lag-1 self edge on the observed time steps (A1). For any `H ≥ 0`, the latent time steps `0, …, H` come before the first observed one.
- The CI answers are separation in the full graph, latent time steps included.
- With links of lag at most `L`, the result holds for every target after time step `L`. With lag-1 links, it holds for every target.
- No initial condition is assumed. Variables of the first observed time step may have latent parents.

## Main theorems

All names are in the namespace `ITPD`, and in `ITPD/Paper.lean` unless a file is given.

- `itpdPaper ci N T`: Algorithm 1. It returns the adjacency list `A`. `u ∈ itpdPaper ci N T ⟨t + 1, n⟩` says that `u` is in the returned `A[Y]` for the target `Y = V^n_{t+1}`.
- `itpd_exact`: abstract CI answers with `TargetHyp` for the target. `A[Y]` is `pa(Y)`.
- `itpd_exact_latent`: separation answers in a graph with latent time steps (`LatentOracle`), and a target with `ParentsObserved`. `A[Y]` is the set of parents of `Y` among the observed variables (`obsE G`).
- `itpd_exact_latent_lag`: links of lag at most `L`. Every target `V^n_{t+1}` with `L ≤ t`.
- `itpd_exact_latent_lag1`: lag-1 links. Every target `V^n_{t+1}` with `1 ≤ t < T` and `1 ≤ n ≤ N`.
- `itpd_exact_nonvacuous`: for every `N`, `T` and `H` there are a graph with the latent time steps `0, …, H` and lag-1 links, and separation answers for it, such that every variable of the first observed time step has a latent parent in every series, and Algorithm 1 returns the parent set of every target.
- The paper's local result is `itpd_exact` and `itpd_exact_latent`. The paper's global result is `itpd_exact_latent_lag1` and `itpd_exact_latent_lag`.
- Seven lemmas hold by `rfl` and show which tests one step of Algorithm 1 asks: `itpdPaper_loops`, `itpdPaper_target`, `itpdPaper_lines_5_6`, `itpdPaper_lines_7_10`, `itpdPaper_line_11`, `itpdPaper_lines_12_13`, `itpdPaper_lines_14_15`.
- The theorems for Algorithm 1 are instances of `itpdS_correct` (`ITPD/Exact.lean`) and `itpdS_correct_latent` (`ITPD/Latent.lean`). These two hold for every order of the candidates in Phase I (Line 6), every order in Phase II (Line 13) and every order of the series (Line 3).

## Assumptions

- A1, lag-1 self edge: the field `LatentDAG.lag1` on the observed time steps. For abstract CI answers: the field `TargetHyp.lag1`.
- A2, no edge inside a time step: the field `LatentDAG.forward` (every edge points strictly forward in time).
- A3, no latent confounding after the first time step: the hypothesis `ParentsObserved G t n` for each target (all parents of the target are observed). `LatentDAG N T H` has the latent time steps `0, …, H` before the `T` observed ones.
- Markov property and faithfulness: not modeled as probability. The CI answers are separation in the graph (`LatentOracle G ci`). The separation relation is `MSep.msep` (`ITPD/Separation.lean`).
- `TargetHyp P ci N t n` (`ITPD/Exact.lean`): all that the proof uses about one target `Y = V^n_{t+1}`. Fields `lag1` (`X → Y`) and `observed` (every parent of `Y` is in `V_{1:t}`) are about the parents. Fields `adj` and `markov` are the two properties of the CI answers that the paper names.
- `adj`: a parent of `Y` is reported dependent on `Y` given every subset of `V_{1:t}` that does not contain it. `markov`: if `pa(Y) ⊆ C ⊆ V_{1:t}`, every `Z` in `V_{1:t}` that is not a parent of `Y` and not in `C` is reported independent of `Y` given `C`.
- `latent_targetHyp` (`ITPD/Latent.lean`): `TargetHyp` follows from `LatentOracle` for every target with `ParentsObserved`.

## Not formalized

- Significance levels and the test statistic. A CI test is its Boolean answer.
- The link from data to CI answers: the Markov property and faithfulness for a probability distribution.
- The equivalence of `MSep.msep` (separation in the moral graph of the ancestral set) with path-based d-separation (Lauritzen, Dawid, Larsen and Leimer, 1990).
- The Python implementation. The proof is about the Lean functions of `ITPD/Algorithm.lean`.
- Line 7 reads `an_hat(S)` from the learned adjacency list `A`. The theorems hold for every learned `A` whose entries are earlier observed variables (`LearnedPast`), so they do not depend on how the learned ancestors at Line 7 are obtained.

## Algorithm 1 line by line

All definitions are in `ITPD/Algorithm.lean`. Its header lists Lines 1-16, and each definition names its lines.

- Line 1: `emptyAdj`, the start of `run`.
- Line 2: `timeSteps T` in `run`.
- Line 3: `runTime`, with the series order `natSeries N`.
- Line 4: `candidates N t ⟨t, n⟩` in `discoverS`.
- Line 5: the start `[x]` of the fold in `phaseIS`.
- Line 6: `descTime`.
- Line 7: `!(inAnHatOfSet A S z)` in `stepIS`.
- Line 8: `ci z y []` in `stepIS`.
- Line 9: `ci z y S` in `stepIS`.
- Line 10: `ci z y (A z) = false` and `insertAll S [z]` in `stepIS`.
- Line 11: `initPa true x S = [x]`.
- Line 12: the start state and `candsII` in `phaseII`.
- Line 13: `ascNode`.
- Line 14: the `else` branch of `stepII`.
- Line 15: the `then` branch of `stepII`.
- Line 16: `update` in `runSeries`.

## Parameters of `itpdS`

- `itpdS ci onlyX givenS candOrd ord2 N T serOrd` (`ITPD/Algorithm.lean`): a more general statement of Algorithm 1, with two Boolean parameters and three orders. Algorithm 1 is one instance of it: `itpdPaper ci N T` is `itpdS ci true true descTime ascNode N T (natSeries N)`.
- `givenS` sets the conditioning set of the Line 9 test. `true` sets `S` (Line 9 of Algorithm 1), and `false` sets `{X}` (`stepIGivenX`, counted only when `Z` and `X` are reported dependent given the empty set). The exactness theorems are proved for both values of `givenS`.
- `onlyX` sets the start of `pa_hat(Y)` in Line 11. `true` sets `{X}` (Line 11 of Algorithm 1), and `false` sets `{X} ∪ (S ∩ V_t)` for the time step `V_t` of `X` (`promote`). The exactness theorems are proved for `onlyX = true` only.
- `discoverS_mem`, `runSeriesS_learnedPast` and `itpdS_learnedPast` (`ITPD/Exact.lean`) hold for both values of `onlyX` and of `givenS`. They say that learned parents are earlier observed variables. They are not exactness theorems.
- `candOrd` (Line 6) and `ord2` (Line 13) are any functions that permute their input (`IsOrder`). `serOrd` (Line 3) is any function with `IsSeriesOrder`.
- `descTime`: descending time, then increasing series index. `ascNode`: ascending time, then increasing series index. `natSeries N`: the series `1, …, N` at every time step. Algorithm 1 states no order inside a time step.

## Axioms

- `#print axioms` on each of the five main theorems gives `propext`, `Classical.choice` and `Quot.sound`. These are the three standard axioms of Lean 4.
- To print them, write a file `Axioms.lean` in this folder with the 7 lines below. Run `lake env lean Axioms.lean` after the build. `Axioms.lean` is not a file of the repository.

```lean
import ITPD
open ITPD
#print axioms itpd_exact
#print axioms itpd_exact_latent
#print axioms itpd_exact_latent_lag
#print axioms itpd_exact_latent_lag1
#print axioms itpd_exact_nonvacuous
```

## Build

- Install Lean with elan. The file `lean-toolchain` pins Lean 4.33.1.
- Run the build command in this folder to compile the 7 modules and the root file `ITPD.lean`:

```
lake build
```

- There is no package dependency. The build downloads nothing once the toolchain is installed.

## Files

- `ITPD/Graph.lean`: vertices, ancestors, the observed past `V_{1:t}`.
- `ITPD/Separation.lean`: the separation relation `MSep.msep`. An edge is never separated. The local Markov property of a target.
- `ITPD/Algorithm.lean`: Algorithm 1 as Lean functions. Definitions only, no theorem.
- `ITPD/Basic.lean`: list facts, the orders of Lines 6 and 13 (`descTime_sorted`, `ascNode_sorted`), the loops of Lines 2-3.
- `ITPD/Exact.lean`: `TargetHyp`. Exactness for abstract CI answers.
- `ITPD/Latent.lean`: graphs with latent time steps (`LatentDAG`). Exactness for separation answers.
- `ITPD/Paper.lean`: `itpdPaper` and the main theorems. Start here.
- `ITPD.lean`: the root file. It imports the 7 modules.
- `lakefile.lean`, `lean-toolchain`, `lake-manifest.json`: build configuration.
