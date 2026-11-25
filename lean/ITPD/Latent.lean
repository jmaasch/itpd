import ITPD.Exact
import ITPD.Separation

/-!
# Latent.lean : exactness with latent time steps before the first observed one

The process runs the time steps `0, …, H` (`H ≥ 0`) before observation begins. The full graph `G.E`
is a graph on all vertices of the process; its time axis counts the latent steps too. The observed
variable `V^n_t` (`1 ≤ t ≤ T`) of the algorithm is the vertex `⟨t + H, n⟩` of the full graph
(`emb`), and every vertex of the full graph with time `≤ H` is latent. The CI answers about
observed variables are the moral-ancestral separation of `Separation.lean` in the full graph,
latent time steps included (`LatentOracle`). No initial condition is assumed: observed variables of
the first step may have latent parents, so they may be latently confounded.

Results:
* `latent_targetHyp`: with these CI answers, every target whose parents in the full graph are
  all observed (`ParentsObserved`) satisfies `TargetHyp`, with the parent relation `obsE` among
  observed variables.
* `itpdS_correct_latent`: Algorithm 1 returns `pa(Y)` at every such target.
* `parentsObserved_of_lag`, `itpdS_correct_latent_lag`, `itpdS_correct_latent_lag1`: with links
  of lag at most `L`, every target after step `L` has only observed parents; with lag-1 links,
  every target.
* `exists_latentOracle`, `fullLatent`: the hypotheses can be met.
-/

set_option autoImplicit false

namespace ITPD

/-- The process graph with the latent steps `0, …, H` before the `T` observed steps, `N` series. -/
structure LatentDAG (N T H : Nat) where
  /-- `E u v`: `u → v` is an edge of the full graph (latent steps included). -/
  E : Vertex → Vertex → Prop
  /-- Every edge points strictly forward in time (contains A2). -/
  forward : ∀ {u v : Vertex}, E u v → u.time < v.time
  /-- Every edge joins two of the `N` series. -/
  series : ∀ {u v : Vertex}, E u v → (1 ≤ u.series ∧ u.series ≤ N) ∧ (1 ≤ v.series ∧ v.series ≤ N)
  /-- A1 on the observed steps: `V^n_t → V^n_{t+1}`. -/
  lag1 : ∀ {t n : Nat}, 1 ≤ t → t < T → 1 ≤ n → n ≤ N → E ⟨t + H, n⟩ ⟨t + 1 + H, n⟩

/-- The observed variable `V^n_t` as a vertex of the full graph. -/
def emb (H : Nat) (v : Vertex) : Vertex := ⟨v.time + H, v.series⟩

/-- The parent relation among observed variables. -/
def obsE {N T H : Nat} (G : LatentDAG N T H) (u v : Vertex) : Prop := G.E (emb H u) (emb H v)

/-- The CI answers about observed variables are moral-ancestral separation in the full graph. -/
def LatentOracle {N T H : Nat} (G : LatentDAG N T H) (ci : CI) : Prop :=
  ∀ (z y : Vertex) (S : List Vertex),
    ci z y S = true ↔ MSep.msep G.E (emb H z) (emb H y) (S.map (emb H))

/-- All parents of `Y = V^n_{t+1}` in the full graph are observed variables (time `> H`). -/
def ParentsObserved {N T H : Nat} (G : LatentDAG N T H) (t n : Nat) : Prop :=
  ∀ w : Vertex, G.E w (emb H ⟨t + 1, n⟩) → H + 1 ≤ w.time

section Model

variable {N T H : Nat} (G : LatentDAG N T H)

theorem LatentDAG.forward' : ∀ u v, G.E u v → u.time < v.time := fun _ _ h => G.forward h

/-- **The hypotheses of `TargetHyp` hold for separation in the full graph**, for every target
`Y = V^n_{t+1}` whose parents in the full graph are observed. Adjacency: one edge is a walk that
no conditioning set blocks. Local Markov: the last step of a walk into `Y` comes from a parent of
`Y`, which is observed and in the conditioning set. -/
theorem latent_targetHyp (ci : CI) (h : LatentOracle G ci) (t n : Nat) (ht : 1 ≤ t)
    (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N) (hobs : ParentsObserved G t n) :
    TargetHyp (obsE G) ci N t n where
  lag1 := by
    have := G.lag1 ht htT hn1 hnN
    simpa [obsE, emb] using this
  observed := by
    intro p hp
    have h1 := hobs _ hp
    have h2 := G.forward hp
    have h3 := (G.series hp).1
    simp [emb] at h1 h2 h3
    exact mem_past.mpr ⟨by omega, by omega, h3.1, h3.2⟩
  adj := by
    intro z S hz _ _
    cases hci : ci z ⟨t + 1, n⟩ S with
    | false => rfl
    | true => exact absurd ((h z _ S).mp hci) (not_msep_of_edge _ hz)
  markov := by
    intro z C hz hzp hzC hC hpa
    have hzt := (mem_past.mp hz).2.1
    apply (h z _ C).mpr
    apply msep_local_markov G.forward'
    · simp [emb]
      omega
    · exact hzp
    · intro s hs
      rcases List.mem_map.mp hs with ⟨c, hc, e⟩
      have := (mem_past.mp (hC c hc)).2.1
      rw [← e]
      simp [emb]
      omega
    · intro w hw
      have hwt := hobs w hw
      have hew : w = emb H ⟨w.time - H, w.series⟩ := by
        rcases w with ⟨wt, ws⟩
        simp only at hwt
        simp [emb]
        omega
      rw [hew] at hw ⊢
      exact List.mem_map.mpr ⟨_, hpa _ hw, rfl⟩

/-- **Links of lag at most `L`**: every target after step `L` has only observed parents. -/
theorem parentsObserved_of_lag (L : Nat) (hlag : ∀ u v, G.E u v → v.time ≤ u.time + L)
    (t n : Nat) (htL : L ≤ t) : ParentsObserved G t n := by
  intro w hw
  have := hlag _ _ hw
  simp [emb] at this
  omega

/-- **Exactness with latent time steps** (`itpdS ci true givenS candOrd ord2 N T serOrd`:
`onlyX = true`, Line 11 of Algorithm 1, both values of `givenS`, every order). Under
`LatentOracle G ci` (separation in the full graph), for `Y = V^n_{t+1}` with `1 ≤ t < T`,
`1 ≤ n ≤ N` and `ParentsObserved G t n`, `A[Y]` is the set of parents of `Y` among observed
variables (`obsE G`). -/
theorem itpdS_correct_latent (ci : CI) (h : LatentOracle G ci) (givenS : Bool)
    (candOrd ord2 : List Vertex → List Vertex) (hc : IsOrder candOrd) (h2 : IsOrder ord2)
    (serOrd : Nat → List Nat) (hser : IsSeriesOrder N serOrd) (t n : Nat) (ht : 1 ≤ t)
    (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N) (hobs : ParentsObserved G t n) :
    ∀ u, u ∈ itpdS ci true givenS candOrd ord2 N T serOrd ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩ :=
  itpdS_correct (obsE G) ci givenS candOrd ord2 hc h2 N T serOrd hser t n htT
    (latent_targetHyp G ci h t n ht htT hn1 hnN hobs)

/-- **Links of lag at most `L`**: exact at every target after step `L`. -/
theorem itpdS_correct_latent_lag (ci : CI) (h : LatentOracle G ci) (givenS : Bool)
    (L : Nat) (hlag : ∀ u v, G.E u v → v.time ≤ u.time + L)
    (candOrd ord2 : List Vertex → List Vertex) (hc : IsOrder candOrd) (h2 : IsOrder ord2)
    (serOrd : Nat → List Nat) (hser : IsSeriesOrder N serOrd) (t n : Nat) (htL : L ≤ t)
    (ht : 1 ≤ t) (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N) :
    ∀ u, u ∈ itpdS ci true givenS candOrd ord2 N T serOrd ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩ :=
  itpdS_correct_latent G ci h givenS candOrd ord2 hc h2 serOrd hser t n ht htT hn1 hnN
    (parentsObserved_of_lag G L hlag t n htL)

/-- **Lag-1 links**: exact at every target `V^n_{t+1}`, `1 ≤ t < T`, `1 ≤ n ≤ N`. -/
theorem itpdS_correct_latent_lag1 (ci : CI) (h : LatentOracle G ci) (givenS : Bool)
    (hlag : ∀ u v, G.E u v → v.time ≤ u.time + 1)
    (candOrd ord2 : List Vertex → List Vertex) (hc : IsOrder candOrd) (h2 : IsOrder ord2)
    (serOrd : Nat → List Nat) (hser : IsSeriesOrder N serOrd) (t n : Nat)
    (ht : 1 ≤ t) (htT : t < T) (hn1 : 1 ≤ n) (hnN : n ≤ N) :
    ∀ u, u ∈ itpdS ci true givenS candOrd ord2 N T serOrd ⟨t + 1, n⟩ ↔ obsE G u ⟨t + 1, n⟩ :=
  itpdS_correct_latent_lag G ci h givenS 1 hlag candOrd ord2 hc h2 serOrd hser t n ht ht
    htT hn1 hnN

end Model

/-! ## The hypotheses can be met -/

/-- The full lag-1 graph on all steps of the process (latent steps included): every series at
step `s` is a parent of every series at step `s + 1`. -/
def fullLatent (N T H : Nat) : LatentDAG N T H where
  E u v := v.time = u.time + 1 ∧ (1 ≤ u.series ∧ u.series ≤ N) ∧ (1 ≤ v.series ∧ v.series ≤ N)
  forward := by
    intro u v h
    omega
  series := fun h => h.2
  lag1 := by
    intro t n _ _ hn1 hnN
    simp
    omega

/-- Separation answers exist for every graph (classical choice of the truth value). -/
theorem exists_latentOracle {N T H : Nat} (G : LatentDAG N T H) : ∃ ci : CI, LatentOracle G ci := by
  refine ⟨fun z y S =>
    @decide (MSep.msep G.E (emb H z) (emb H y) (S.map (emb H))) (Classical.propDecidable _), ?_⟩
  intro z y S
  exact @decide_eq_true_iff _ (Classical.propDecidable _)

end ITPD
