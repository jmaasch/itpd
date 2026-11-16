import ITPD.Latent

/-!
# Paper.lean : the algorithm box of the paper and its exactness theorems

`itpdPaper` is Algorithm 1 (ITPD) as in the box of the paper: the test of Line 9 is given `S`,
Line 11 is `pa_hat(Y) ← {X}`, the candidates of Line 6 are visited in descending time, the
members of `Z'` in Line 13 in ascending time, and the series of Line 3 in the order `1, …, N`.
Inside a time step both orders visit the series in increasing index; the box does not state an
order inside a time step, and the general theorems of `Exact.lean` and `Latent.lean` hold for
every order.

The first lemmas hold by `rfl` and show which CI tests one step of the box asks. The exactness
theorems below have no switch and no order parameter:

* `itpd_exact`: for CI answers with the two properties of `TargetHyp` for the target `Y`,
  `A[Y] = pa(Y)`.
* `itpd_exact_latent`: for CI answers given by separation in a graph that may have latent time
  steps before the first observed one, `A[Y] = pa(Y)` for every target `Y` whose parents are all
  observed.
* `itpd_exact_latent_lag`, `itpd_exact_latent_lag1`: with links of lag at most `L`, every target
  after step `L`; with lag-1 links, every target.
* `itpd_exact_nonvacuous`: the hypotheses of `itpd_exact_latent_lag1` can be met, with latent
  parents of every variable of the first observed step.
-/

set_option autoImplicit false

namespace ITPD

/-- **Algorithm 1 (ITPD) as in the box of the paper.** Returns the adjacency list `A`. -/
def itpdPaper (ci : CI) (N T : Nat) : Adj :=
  itpdS ci true true descTime ascNode N T (natSeries N)

/-! ## What the box computes (all by `rfl`) -/

/-- Lines 1-3 and 16: the box is the loop over the time steps `1, …, T - 1` and the series
`1, …, N`, with Lines 4-15 as the body. -/
theorem itpdPaper_loops (ci : CI) (N T : Nat) :
    itpdPaper ci N T =
      run (fun A t n => discoverS ci true true descTime ascNode N A t n) T (natSeries N) := rfl

/-- Lines 4-15 for the target `Y = V^n_{t+1}`: Phase I over the candidates in descending time,
then Phase II. -/
theorem itpdPaper_target (ci : CI) (N : Nat) (A : Adj) (t n : Nat) :
    discoverS ci true true descTime ascNode N A t n =
      phaseII ci ascNode true ⟨t, n⟩ ⟨t + 1, n⟩
        (phaseIS ci true A ⟨t, n⟩ ⟨t + 1, n⟩ (descTime (candidates N t ⟨t, n⟩))) := rfl

/-- Lines 5-6: Phase I starts from `S = {X}` and visits the candidates one at a time. -/
theorem itpdPaper_lines_5_6 (ci : CI) (A : Adj) (x y : Vertex) (cands : List Vertex) :
    phaseIS ci true A x y cands = cands.foldl (stepIS ci true A x y) [x] := rfl

/-- Lines 7-10 for one candidate `Z`: the zero-order test of Line 8 (asked when `Z` is not a
learned ancestor of a member of `S`, Line 7), the test given `S` of Line 9, and the local Markov
test given the learned parents of `Z` of Line 10. `Z` enters `S` only when Line 10 reports
dependence. -/
theorem itpdPaper_lines_7_10 (ci : CI) (A : Adj) (x y : Vertex) (S : List Vertex) (z : Vertex) :
    stepIS ci true A x y S z =
      if (!(inAnHatOfSet A S z) && ci z y []) = true then S
      else if ci z y S = true then S
      else if ci z y (A z) = false then insertAll S [z]
      else S := rfl

/-- Line 11: `pa_hat(Y) ← {X}`. -/
theorem itpdPaper_line_11 (x : Vertex) (S : List Vertex) : initPa true x S = [x] := rfl

/-- Lines 12-13: Phase II starts from the state (`pa_hat(Y) = {X}`, `S' = S`) and visits
`Z' = S \ pa_hat(Y)` in ascending time. -/
theorem itpdPaper_lines_12_13 (ci : CI) (x y : Vertex) (S : List Vertex) :
    phaseII ci ascNode true x y S =
      ((ascNode (S.filter (fun z => decide (z ∉ [x])))).foldl (stepII ci y) ([x], S)).1 := rfl

/-- Lines 14-15 for one `Z`: the test given `S' \ {Z}`; dependent: `Z` is a parent; independent:
`Z` is removed from `S'`. -/
theorem itpdPaper_lines_14_15 (ci : CI) (y : Vertex) (pa S' : List Vertex) (z : Vertex) :
    stepII ci y (pa, S') z =
      if ci z y (S'.filter (fun s => s != z)) = true
      then (pa, S'.filter (fun s => s != z))
      else (pa ++ [z], S') := rfl

/-! ## Exactness of the box -/

/-- **Exactness of Algorithm 1 for one target, abstract CI answers.** If the CI answers have the
two properties of `TargetHyp` for the target `Y = V^n_{t+1}` (`t < T`) and the parent relation
`P`, the returned entry `A[Y]` is `pa(Y)`. -/
theorem itpd_exact (P : Vertex → Vertex → Prop) (ci : CI) (N T t n : Nat) (htT : t < T)
    (H : TargetHyp P ci N t n) :
    ∀ u, u ∈ itpdPaper ci N T ⟨t + 1, n⟩ ↔ P u ⟨t + 1, n⟩ :=
  itpdS_correct P ci true descTime ascNode isOrder_descTime isOrder_ascNode N T (natSeries N)
    (isSeriesOrder_natSeries N) t n htT H

/-- **Exactness of Algorithm 1 with latent time steps.** Let `G` be a graph with `H ≥ 0` latent
time steps before the `T` observed ones, with edges forward in time and the lag-1 self edge of
every series on the observed steps (A1), and let the CI answers be separation in the full graph
`G`. For every target `Y = V^n_{t+1}` (`1 ≤ t < T`, `1 ≤ n ≤ N`) whose parents are all observed,
the returned entry `A[Y]` is the parent set of `Y`. -/
theorem itpd_exact_latent {N T H : Nat} (G : LatentDAG N T H) (ci : CI) (h : LatentOracle G ci)
    (t n : Nat) (ht : 1 ≤ t) (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N)
    (hobs : ParentsObserved G t n) :
    ∀ u, u ∈ itpdPaper ci N T ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩ :=
  itpdS_correct_latent G ci h true descTime ascNode isOrder_descTime isOrder_ascNode
    (natSeries N) (isSeriesOrder_natSeries N) t n ht htT hn1 hnN hobs

/-- **Links of lag at most `L`**: Algorithm 1 is exact at every target after step `L`. -/
theorem itpd_exact_latent_lag {N T H : Nat} (G : LatentDAG N T H) (ci : CI)
    (h : LatentOracle G ci) (L : Nat) (hlag : ∀ u v, G.E u v → v.time ≤ u.time + L)
    (t n : Nat) (htL : L ≤ t) (ht : 1 ≤ t) (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N) :
    ∀ u, u ∈ itpdPaper ci N T ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩ :=
  itpdS_correct_latent_lag G ci h true L hlag descTime ascNode isOrder_descTime isOrder_ascNode
    (natSeries N) (isSeriesOrder_natSeries N) t n htL ht htT hn1 hnN

/-- **Lag-1 links**: Algorithm 1 is exact at every target `V^n_{t+1}`, `1 ≤ t < T`,
`1 ≤ n ≤ N`. -/
theorem itpd_exact_latent_lag1 {N T H : Nat} (G : LatentDAG N T H) (ci : CI)
    (h : LatentOracle G ci) (hlag : ∀ u v, G.E u v → v.time ≤ u.time + 1)
    (t n : Nat) (ht : 1 ≤ t) (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N) :
    ∀ u, u ∈ itpdPaper ci N T ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩ :=
  itpdS_correct_latent_lag1 G ci h true hlag descTime ascNode isOrder_descTime isOrder_ascNode
    (natSeries N) (isSeriesOrder_natSeries N) t n ht htT hn1 hnN

/-- **The hypotheses can be met with latent parents at the first observed step.** For every `N`,
`T` and `H` there are a graph with `H` latent steps and lag-1 links, and separation answers for
it, such that (1) every variable of the first observed step has a parent in every series one
step earlier, which is a latent vertex and not an observed variable, and (2) Algorithm 1 returns
the parent set of every target `V^n_{t+1}`, `1 ≤ t < T`, `1 ≤ n ≤ N`. -/
theorem itpd_exact_nonvacuous (N T H : Nat) :
    ∃ (G : LatentDAG N T H) (ci : CI), LatentOracle G ci ∧
      (∀ u v, G.E u v → v.time ≤ u.time + 1) ∧
      (∀ n m, 1 ≤ n → n ≤ N → 1 ≤ m → m ≤ N →
        G.E ⟨H, m⟩ (emb H ⟨1, n⟩) ∧ ∀ u : Vertex, 1 ≤ u.time → emb H u ≠ ⟨H, m⟩) ∧
      (∀ t n, 1 ≤ t → t < T → 1 ≤ n → n ≤ N →
        ∀ u, u ∈ itpdPaper ci N T ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩) := by
  rcases exists_latentOracle (fullLatent N T H) with ⟨ci, h⟩
  have hlag : ∀ u v, (fullLatent N T H).E u v → v.time ≤ u.time + 1 := by
    intro u v hv
    simp [fullLatent] at hv
    omega
  refine ⟨fullLatent N T H, ci, h, hlag, ?_, ?_⟩
  · intro n m hn1 hnN hm1 hmN
    refine ⟨?_, ?_⟩
    · simp [fullLatent, emb]
      omega
    · intro u hu e
      simp [emb] at e
      omega
  · intro t n ht htT hn1 hnN
    exact itpd_exact_latent_lag1 (fullLatent N T H) ci h hlag t n ht htT hn1 hnN

end ITPD
