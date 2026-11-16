# Lean 4 proof: ITPD returns the parent set

This folder holds a machine-checked proof that Algorithm 1 of the paper (ITPD) is exact when the
conditional independence (CI) answers are those of the graph. It uses Lean 4.33.1 and the Lean
core library only (no Mathlib, no other dependency).

## What is proved

Algorithm 1, as in the box of the paper (Line 9: test of `(Z, Y)` given `S`; Line 10: local
Markov test given the learned parents of `Z`; Line 11: `pa_hat(Y) <- {X}`; candidates in
descending time; Phase II in ascending time), returns `A[Y] = pa(Y)` for every target `Y` whose
parents are all observed, when

- the graph has edges that point forward in time and the lag-1 self edge of every series
  (A1); it may have any number of latent time steps before the first observed one, and
- the CI answers are d-separation in the full graph, latent time steps included.

With links of lag at most `L`, every target after time step `L` qualifies; with lag-1 links,
every target does. No initial condition is assumed: the variables of the first observed time
step may have latent parents.

Every proof is complete, and the library declares no axiom. The theorems depend only on the
three standard axioms of Lean (see "Axioms").

## Main theorems

All are in `ITPD/Paper.lean`, in the namespace `ITPD`. `itpdPaper ci N T` is the box:

```lean
def itpdPaper (ci : CI) (N T : Nat) : Adj :=
  itpdS ci true true descTime ascNode N T (natSeries N)
```

(a) Abstract CI answers with two properties for the target (`TargetHyp`, below):

```lean
theorem itpd_exact (P : Vertex → Vertex → Prop) (ci : CI) (N T t n : Nat) (htT : t < T)
    (H : TargetHyp P ci N t n) :
    ∀ u, u ∈ itpdPaper ci N T ⟨t + 1, n⟩ ↔ P u ⟨t + 1, n⟩
```

(b) CI answers given by separation in a graph with `H ≥ 0` latent time steps:

```lean
theorem itpd_exact_latent {N T H : Nat} (G : LatentDAG N T H) (ci : CI) (h : LatentOracle G ci)
    (t n : Nat) (ht : 1 ≤ t) (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N)
    (hobs : ParentsObserved G t n) :
    ∀ u, u ∈ itpdPaper ci N T ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩

theorem itpd_exact_latent_lag {N T H : Nat} (G : LatentDAG N T H) (ci : CI)
    (h : LatentOracle G ci) (L : Nat) (hlag : ∀ u v, G.E u v → v.time ≤ u.time + L)
    (t n : Nat) (htL : L ≤ t) (ht : 1 ≤ t) (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N) :
    ∀ u, u ∈ itpdPaper ci N T ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩

theorem itpd_exact_latent_lag1 {N T H : Nat} (G : LatentDAG N T H) (ci : CI)
    (h : LatentOracle G ci) (hlag : ∀ u v, G.E u v → v.time ≤ u.time + 1)
    (t n : Nat) (ht : 1 ≤ t) (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N) :
    ∀ u, u ∈ itpdPaper ci N T ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩
```

Here `⟨t + 1, n⟩` is the target `Y = V^n_{t+1}`, and `u ∈ itpdPaper ci N T ⟨t + 1, n⟩` says that
`u` is in the returned `A[Y]`.

`itpd_exact_nonvacuous` shows that the hypotheses of `itpd_exact_latent_lag1` can be met by a
graph in which every variable of the first observed time step has latent parents.

The same file has lemmas that hold by `rfl` and show which tests one step of the box asks:
`itpdPaper_loops`, `itpdPaper_target`, `itpdPaper_lines_5_6`, `itpdPaper_lines_7_10`,
`itpdPaper_line_11`, `itpdPaper_lines_12_13`, `itpdPaper_lines_14_15`. For example:

```lean
theorem itpdPaper_lines_7_10 (ci : CI) (A : Adj) (x y : Vertex) (S : List Vertex) (z : Vertex) :
    stepIS ci true A x y S z =
      if (!(inAnHatOfSet A S z) && ci z y []) = true then S
      else if ci z y S = true then S
      else if ci z y (A z) = false then insertAll S [z]
      else S

theorem itpdPaper_line_11 (x : Vertex) (S : List Vertex) : initPa true x S = [x]
```

The theorems for the box are instances of more general ones: `itpdS_correct` (`ITPD/Exact.lean`)
and `itpdS_correct_latent` (`ITPD/Latent.lean`) hold for every candidate order (Line 6), every
Phase II order (Line 13) and every series order (Line 3).

## Assumptions

| Paper | Lean |
| --- | --- |
| A1, lag-1 self edge | `TargetHyp.lag1`; `LatentDAG.lag1` (on the observed time steps) |
| A2, no edge inside a time step | `LatentDAG.forward`: every edge points strictly forward in time |
| Latent confounding only at the first observed time step | `LatentDAG N T H` has `H` latent time steps before the observed ones; the theorem is for each target with `ParentsObserved G t n` (all parents of the target are observed) |
| Markov property and faithfulness | not modeled as probability; the CI answers are separation in the graph (`LatentOracle`) |

`TargetHyp P ci N t n` (`ITPD/Exact.lean`) is all that the proof uses about the CI answers for
the target `Y = V^n_{t+1}`:

- `lag1`: `X → Y`.
- `observed`: every parent of `Y` is in `V_{1:t}`.
- `adj`: a parent of `Y` is reported dependent on `Y` given every subset of `V_{1:t}` that does
  not contain it.
- `markov`: if `pa(Y) ⊆ C ⊆ V_{1:t}`, every `Z` in `V_{1:t}` that is not a parent of `Y` and not
  in `C` is reported independent of `Y` given `C`.

`latent_targetHyp` (`ITPD/Latent.lean`) proves `TargetHyp` from `LatentOracle` for every target
with `ParentsObserved`.

## What is not formalized

- **Significance levels.** A CI test is its Boolean answer. The levels of the box and the test
  statistic are not modeled.
- **The link to data.** The Markov property and faithfulness are not stated for a probability
  distribution. The theorems assume that the CI answers are the separation statements of the
  graph.
- **Path-based d-separation.** `MSep.msep` (`ITPD/Separation.lean`) is separation in the moral
  graph of the ancestral set. Its equivalence with path-based d-separation (Lauritzen, Dawid,
  Larsen and Leimer, 1990) is not proved here.
- **The Python code.** The proof is about the Lean functions of `ITPD/Algorithm.lean`. It does
  not verify the implementation.
- **The latent-confounding assumption** has the form "all parents of the target are observed"
  (`ParentsObserved`), for each target. `itpd_exact_latent_lag1` gives every target under lag-1
  links.
- **Line 7** reads `an_hat(S)` from the learned adjacency list `A` only. The theorems hold for
  every learned `A`, so they do not depend on how `an_hat` is obtained.
- **Order inside a time step.** The box does not state one. `descTime` and `ascNode` visit the
  series of a time step in increasing index. The general theorems hold for every order.

## Paper line to Lean definition

All definitions are in `ITPD/Algorithm.lean`.

| Line | Step in the box | Lean |
| --- | --- | --- |
| 1 | `A <- {V^i_1 -> {}}` | `emptyAdj`, start of `run` |
| 2-3 | loops over `t` and `n` | `timeSteps T`, `runTime`; series order `natSeries N` |
| 4 | `X`, `Y`, `Z = V_{1:t} \ {X, Y}` | `candidates N t ⟨t, n⟩` in `discoverS` |
| 5 | `S <- {X}` | start `[x]` of the fold in `phaseIS` |
| 6 | for `Z`, descending time | `descTime` (`descTime_sorted`) |
| 7 | if `Z` not in `an_hat(S)` | `!(inAnHatOfSet A S z)` in `stepIS` |
| 8 | if `CI(Z, Y)` independent: continue | `ci z y []` in `stepIS` |
| 9 | if `CI(Z, Y given S)` independent: continue | `ci z y S` in `stepIS` |
| 10 | if `CI(Z, Y given pa_hat(Z))` dependent: `S <- S + {Z}` | `ci z y (A z) = false`, `insertAll S [z]` in `stepIS` |
| 11 | `pa_hat(Y) <- {X}` | `initPa true x S = [x]` |
| 12 | `S' <- S`, `Z' <- S \ pa_hat(Y)` | start state `(initPa, S)` and `candsII` in `phaseII` |
| 13 | for `Z` in `Z'`, ascending time | `ascNode` (`ascNode_sorted`) |
| 14 | if `CI(Z, Y given S' \ {Z})` dependent: `Z` is a parent | `stepII`, second branch |
| 15 | else `S' <- S' \ {Z}` | `stepII`, first branch |
| 16 | `A[Y] <- pa_hat(Y)` | `update` in `runSeries` |

Notes on names:

- `itpdS ci onlyX givenS candOrd ord2 N T serOrd` is Algorithm 1 with two Boolean switches and
  three order parameters. `givenS = true` is Line 9 of the box and `onlyX = true` is Line 11 of
  the box. The value `false` of a switch selects an alternative step that is not in the paper
  (`stepIGivenX`, `promote`). The exactness theorems fix `onlyX = true` and hold for both values
  of `givenS`; `itpdPaper` fixes both switches to `true`.
- `ascNode` is the order "ascending time, then increasing series index".
- `obsE G u v` is the parent relation of `G` between observed variables.

## Files

| File | Content |
| --- | --- |
| `ITPD/Graph.lean` | vertices, ancestors, the observed past `V_{1:t}` |
| `ITPD/Separation.lean` | the separation relation; an edge is never separated; local Markov property of a target |
| `ITPD/Algorithm.lean` | Algorithm 1 as Lean functions |
| `ITPD/Basic.lean` | list facts, the orders of Lines 6 and 13, the loops of Lines 2-3 |
| `ITPD/Exact.lean` | `TargetHyp`, exactness for abstract CI answers |
| `ITPD/Latent.lean` | graphs with latent time steps, exactness for separation answers |
| `ITPD/Paper.lean` | `itpdPaper` and the theorems for the box |

## Axioms

Output of `#print axioms`:

```
'ITPD.itpd_exact' depends on axioms: [propext, Classical.choice, Quot.sound]
'ITPD.itpd_exact_latent' depends on axioms: [propext, Classical.choice, Quot.sound]
'ITPD.itpd_exact_latent_lag' depends on axioms: [propext, Classical.choice, Quot.sound]
'ITPD.itpd_exact_latent_lag1' depends on axioms: [propext, Classical.choice, Quot.sound]
'ITPD.itpd_exact_nonvacuous' depends on axioms: [propext, Classical.choice, Quot.sound]
```

These are the three standard axioms of Lean 4. To print them again, put the lines below in a
file `Axioms.lean` in this folder and run `lake env lean Axioms.lean` after the build.

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

Install Lean with elan (see https://lean-lang.org). The file `lean-toolchain` pins Lean 4.33.1.
Then, in this folder:

```
lake build
```

The build compiles 7 modules and the root `ITPD.lean` in a few seconds. The project has no
package dependency, so `lake build` downloads nothing once the toolchain is installed.
