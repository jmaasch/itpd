import ITPD.Basic

/-!
# Exact.lean : Algorithm 1 returns `pa(Y)` for every target that meets `TargetHyp`

The CI answers are an abstract function with two properties for the target `Y` (`TargetHyp`):
a parent of `Y` is reported dependent on `Y` given every set of observed earlier variables that
does not contain it, and `Y` has the local Markov property with weak union. The parent relation
`P` is any relation on the observed variables; `Latent.lean` takes the parent relation of a
graph that may have latent time steps before the first observed one.

Main result, `itpdS_correct`: for a target `Y` that meets `TargetHyp`,
`itpdS ci true givenS candOrd ord2 N T serOrd` returns the entry `A[Y]` equal to `pa(Y)`. This holds
with `onlyX = true` (Line 11 of Algorithm 1), for both values of the Boolean parameter `givenS`,
every candidate order (Line 6), every Phase II order (Line 13), and every series order (Line 3). No
hypothesis is made on other targets, on the first time step, or on the correctness of the learned
graph at earlier steps.

Outline of the proof.
* Phase I keeps every parent (`phaseIS_parents`): each of the tests of Lines 8, 9 and 10 reports
  dependence for a parent that is not yet in `S`. So `pa(Y) ⊆ S ⊆ V_{1:t}` after Line 10.
* Phase II removes every non-parent and keeps every parent (`foldII_correct`): when `Z` is
  visited, `S' \ {Z}` contains `pa(Y) \ {Z}`, so Line 14 reports dependence exactly for parents.
* The learned graph is used only through `LearnedPast` (learned parents are earlier observed
  variables), which holds for arbitrary CI answers (`itpdS_learnedPast`).
-/

set_option autoImplicit false

namespace ITPD

/-! ## The hypotheses for one target -/

/-- The hypotheses for the target `Y = V^n_{t+1}` with `X = V^n_t`, for a parent relation `P` on
the observed variables (`P u v`: `u` is a parent of `v`).
* `lag1`: `X → Y` (assumption A1).
* `observed`: every parent of `Y` is an observed variable of `V_{1:t}`.
* `adj`: a parent of `Y` is reported dependent on `Y` given every set of observed variables of
  `V_{1:t}` that does not contain it (adjacency faithfulness).
* `markov`: if `pa(Y) ⊆ C ⊆ V_{1:t}` and `Z ∈ V_{1:t}` is not a parent of `Y` and not in `C`, then
  `Z` is reported independent of `Y` given `C` (local Markov property of `Y` and weak union). -/
structure TargetHyp (P : Vertex → Vertex → Prop) (ci : CI) (N t n : Nat) : Prop where
  lag1 : P ⟨t, n⟩ ⟨t + 1, n⟩
  observed : ∀ p, P p ⟨t + 1, n⟩ → p ∈ past N t
  adj : ∀ (z : Vertex) (S : List Vertex), P z ⟨t + 1, n⟩ → (∀ s, s ∈ S → s ∈ past N t) →
    z ∉ S → ci z ⟨t + 1, n⟩ S = false
  markov : ∀ (z : Vertex) (C : List Vertex), z ∈ past N t → ¬ P z ⟨t + 1, n⟩ → z ∉ C →
    (∀ c, c ∈ C → c ∈ past N t) → (∀ p, P p ⟨t + 1, n⟩ → p ∈ C) → ci z ⟨t + 1, n⟩ C = true

/-- Learned parents are earlier observed variables. Every adjacency list that the algorithm
builds has this property, for arbitrary CI answers (`itpdS_learnedPast`). -/
def LearnedPast (N : Nat) (A : Adj) : Prop :=
  ∀ v u : Vertex, u ∈ A v → 1 ≤ u.time ∧ u.time < v.time ∧ 1 ≤ u.series ∧ u.series ≤ N

theorem learnedPast_empty (N : Nat) : LearnedPast N emptyAdj := by
  intro v u hu
  exact absurd hu List.not_mem_nil

/-- A learned parent of a variable of `V_{1:t}` is another variable of `V_{1:t}`. -/
theorem LearnedPast.past_of_mem {N : Nat} {A : Adj} (hA : LearnedPast N A) {t : Nat} {z u : Vertex}
    (hz : z ∈ past N t) (hu : u ∈ A z) : u ∈ past N t ∧ u ≠ z := by
  have h1 := hA z u hu
  have h2 := mem_past.mp hz
  refine ⟨mem_past.mpr ⟨h1.1, by omega, h1.2.2.1, h1.2.2.2⟩, ?_⟩
  intro e
  rw [e] at h1
  omega

/-! ## Phase I and Phase II for arbitrary CI answers -/

/-- `stepIGivenX` (Lines 7-10 for `givenS = false`) keeps `S` or adds `Z` to it, adding `Z` only
when the Line 10 test reports dependence. -/
theorem stepIGivenX_cases (ci : CI) (A : Adj) (x y : Vertex) (S : List Vertex) (z : Vertex) :
    stepIGivenX ci A x y S z = S ∨
      (stepIGivenX ci A x y S z = insertAll S [z] ∧ ci z y (A z) = false) := by
  unfold stepIGivenX
  split
  · exact Or.inl rfl
  · split
    · exact Or.inl rfl
    · split
      · rename_i h
        exact Or.inr ⟨rfl, h⟩
      · exact Or.inl rfl

theorem stepIGivenX_sub (ci : CI) (A : Adj) (x y : Vertex) (S : List Vertex) (z : Vertex) :
    ∀ s, s ∈ stepIGivenX ci A x y S z ↔ s ∈ S ∨ (s = z ∧ stepIGivenX ci A x y S z ≠ S) := by
  intro s
  rcases stepIGivenX_cases ci A x y S z with h | ⟨h, _⟩
  · rw [h]
    simp
  · rw [h, mem_insertAll]
    by_cases hs : s ∈ S
    · simp [hs]
    · simp [hs]
      intro e
      subst e
      intro h2
      exact hs (h2 ▸ (mem_insertAll S [s] s).mpr (Or.inr (List.mem_singleton.mpr rfl)))

/-- Lines 7-10 keep `S` or add `Z`; `Z` is added only when Line 10 reports dependence. -/
theorem stepIS_cases (ci : CI) (givenS : Bool) (A : Adj) (x y : Vertex) (S : List Vertex)
    (z : Vertex) :
    stepIS ci givenS A x y S z = S ∨
      (stepIS ci givenS A x y S z = insertAll S [z] ∧ ci z y (A z) = false) := by
  cases givenS with
  | false => exact stepIGivenX_cases ci A x y S z
  | true =>
    simp only [stepIS]
    split
    · exact Or.inl rfl
    · split
      · exact Or.inl rfl
      · split
        · rename_i h
          exact Or.inr ⟨rfl, h⟩
        · exact Or.inl rfl

/-- Members of `S` after Lines 7-10 for the candidate `z`. -/
theorem stepIS_sub (ci : CI) (givenS : Bool) (A : Adj) (x y : Vertex) (S : List Vertex)
    (z : Vertex) :
    ∀ s, s ∈ stepIS ci givenS A x y S z ↔ s ∈ S ∨ (s = z ∧ stepIS ci givenS A x y S z ≠ S) := by
  intro s
  rcases stepIS_cases ci givenS A x y S z with h | ⟨h, _⟩
  · rw [h]
    simp
  · rw [h, mem_insertAll]
    by_cases hs : s ∈ S
    · simp [hs]
    · simp [hs]
      intro e
      subst e
      intro h2
      exact hs (h2 ▸ (mem_insertAll S [s] s).mpr (Or.inr (List.mem_singleton.mpr rfl)))

/-- Phase I adds only candidates. -/
theorem foldIS_mem (ci : CI) (givenS : Bool) (A : Adj) (x y : Vertex) :
    ∀ (cands S : List Vertex) (s : Vertex), s ∈ cands.foldl (stepIS ci givenS A x y) S →
      s ∈ S ∨ s ∈ cands
  | [], _, _, h => Or.inl h
  | z :: cands, S, s, h => by
    rw [List.foldl_cons] at h
    rcases foldIS_mem ci givenS A x y cands _ s h with h | h
    · rcases (stepIS_sub ci givenS A x y S z s).mp h with h | ⟨h, _⟩
      · exact Or.inl h
      · exact Or.inr (List.mem_cons.mpr (Or.inl h))
    · exact Or.inr (List.mem_cons.mpr (Or.inr h))

/-- `stepII` (Lines 14-15) adds only the visited `Z` to `pa_hat(Y)`. -/
theorem foldII_mem (ci : CI) (y : Vertex) :
    ∀ (L : List Vertex) (st : List Vertex × List Vertex) (u : Vertex),
      u ∈ (L.foldl (stepII ci y) st).1 → u ∈ st.1 ∨ u ∈ L
  | [], _, _, h => Or.inl h
  | z :: L, st, u, h => by
    rw [List.foldl_cons] at h
    rcases foldII_mem ci y L _ u h with h | h
    · unfold stepII at h
      split at h
      · exact Or.inl h
      · rcases List.mem_append.mp h with h | h
        · exact Or.inl h
        · exact Or.inr (List.mem_cons.mpr (Or.inl (List.mem_singleton.mp h)))
    · exact Or.inr (List.mem_cons.mpr (Or.inr h))

/-- Every member of the output of Lines 4-15 is `X` or a member of `V_{1:t}`, for arbitrary CI
answers, both values of `onlyX`, both values of `givenS`, and all orders `candOrd` and `ord2` with
`IsOrder`. -/
theorem discoverS_mem (ci : CI) (onlyX givenS : Bool) (candOrd ord2 : List Vertex → List Vertex)
    (hc : IsOrder candOrd) (h2 : IsOrder ord2) (N : Nat) (A : Adj) (t n : Nat) (u : Vertex)
    (hu : u ∈ discoverS ci onlyX givenS candOrd ord2 N A t n) :
    u = ⟨t, n⟩ ∨ u ∈ past N t := by
  unfold discoverS phaseII at hu
  have hS : ∀ s, s ∈ phaseIS ci givenS A ⟨t, n⟩ ⟨t + 1, n⟩ (candOrd (candidates N t ⟨t, n⟩)) →
      s = ⟨t, n⟩ ∨ s ∈ past N t := by
    intro s hs
    unfold phaseIS at hs
    rcases foldIS_mem ci givenS A _ _ _ _ s hs with h | h
    · exact Or.inl (List.mem_singleton.mp h)
    · exact Or.inr ((mem_candidates N t _ s).mp ((hc.mem_iff _ s).mp h)).1
  rcases foldII_mem ci _ _ _ u hu with h | h
  · unfold initPa at h
    split at h
    · exact Or.inl (List.mem_singleton.mp h)
    · unfold promote at h
      rcases List.mem_cons.mp h with h | h
      · exact Or.inl h
      · exact hS u (List.mem_filter.mp h).1
  · rw [h2.mem_iff] at h
    unfold candsII at h
    exact hS u (List.mem_filter.mp h).1

/-- One execution of Lines 4-16 keeps "learned parents are earlier observed variables". -/
theorem runSeriesS_learnedPast (ci : CI) (onlyX givenS : Bool)
    (candOrd ord2 : List Vertex → List Vertex) (hc : IsOrder candOrd) (h2 : IsOrder ord2)
    (N : Nat) (B : Adj) (s m : Nat) (hs1 : 1 ≤ s) (hm : 1 ≤ m ∧ m ≤ N) (hB : LearnedPast N B) :
    LearnedPast N
      (runSeries (fun A t n => discoverS ci onlyX givenS candOrd ord2 N A t n) s B m) := by
  intro v u hu
  unfold runSeries update at hu
  split at hu
  · rename_i hv
    subst hv
    rcases discoverS_mem ci onlyX givenS candOrd ord2 hc h2 N B s m u hu with h | h
    · subst h
      simp
      omega
    · have := mem_past.mp h
      simp
      omega
  · exact hB v u hu

/-- **Learned parents are earlier observed variables** (`LearnedPast` on the adjacency list that
`itpdS` returns), for arbitrary CI answers, both values of `onlyX`, both values of `givenS`, all
orders with `IsOrder` and every series order with `IsSeriesOrder`. -/
theorem itpdS_learnedPast (ci : CI) (onlyX givenS : Bool)
    (candOrd ord2 : List Vertex → List Vertex) (hc : IsOrder candOrd) (h2 : IsOrder ord2)
    (N T : Nat) (serOrd : Nat → List Nat) (hser : IsSeriesOrder N serOrd) :
    LearnedPast N (itpdS ci onlyX givenS candOrd ord2 N T serOrd) := by
  have key : ∀ (Ts : List Nat) (A : Adj), (∀ s, s ∈ Ts → 1 ≤ s) → LearnedPast N A →
      LearnedPast N (Ts.foldl
        (runTime (fun A t n => discoverS ci onlyX givenS candOrd ord2 N A t n) serOrd) A) := by
    intro Ts
    induction Ts with
    | nil => intro A _ hA; exact hA
    | cons s Ts ih =>
      intro A hTs hA
      rw [List.foldl_cons]
      apply ih _ (fun s' hs' => hTs s' (List.mem_cons.mpr (Or.inr hs')))
      have hs1 : 1 ≤ s := hTs s (List.mem_cons.mpr (Or.inl rfl))
      apply foldSeries_keep _ s (LearnedPast N) (serOrd s) A _ hA
      intro B m hm hB
      exact runSeriesS_learnedPast ci onlyX givenS candOrd ord2 hc h2 N B s m hs1
        ((mem_serOrd hser s m).mp hm) hB
  exact key (timeSteps T) emptyAdj (fun _ hs => (mem_timeSteps.mp hs).1) (learnedPast_empty N)

/-! ## Exactness for one target -/

section OneTarget

variable {P : Vertex → Vertex → Prop} {ci : CI} {N t n : Nat}

/-- Under `TargetHyp`, with `X ∈ S` and `LearnedPast N A`, a parent `p` of `Y` is in `S` after
`stepIGivenX` visits `p` (a parent not yet in `S` is added). Case `givenS = false` of
`stepIS_parent`. -/
theorem stepIGivenX_parent (H : TargetHyp P ci N t n) (A : Adj) (hA : LearnedPast N A)
    (S : List Vertex) (hx : (⟨t, n⟩ : Vertex) ∈ S)
    (p : Vertex) (hp : P p ⟨t + 1, n⟩) :
    p ∈ stepIGivenX ci A ⟨t, n⟩ ⟨t + 1, n⟩ S p := by
  by_cases hpS : p ∈ S
  · exact (stepIGivenX_sub ci A _ _ S p p).mpr (Or.inl hpS)
  · have hpp := H.observed p hp
    have hpx : p ≠ (⟨t, n⟩ : Vertex) := fun e => hpS (e ▸ hx)
    have h8 : ci p ⟨t + 1, n⟩ [] = false :=
      H.adj p [] hp (fun _ hs => absurd hs List.not_mem_nil) List.not_mem_nil
    have h9 : ci p ⟨t + 1, n⟩ [⟨t, n⟩] = false := by
      apply H.adj p _ hp
      · intro s hs
        rw [List.mem_singleton.mp hs]
        exact H.observed _ H.lag1
      · intro h
        exact hpx (List.mem_singleton.mp h)
    have h10 : ci p ⟨t + 1, n⟩ (A p) = false :=
      H.adj p (A p) hp (fun s hs => (hA.past_of_mem hpp hs).1) (fun h => (hA.past_of_mem hpp h).2 rfl)
    have e : stepIGivenX ci A ⟨t, n⟩ ⟨t + 1, n⟩ S p = insertAll S [p] := by
      unfold stepIGivenX
      simp [h8, h9, h10]
    rw [e]
    exact (mem_insertAll S [p] p).mpr (Or.inr (List.mem_singleton.mpr rfl))

/-- A parent of `Y` that is not yet in `S` is added by Lines 7-10, for both values of `givenS`. For
`givenS = true`, the test of Line 9 is given `S`; `S ⊆ V_{1:t}` and the parent is not in `S`, so
`adj` reports dependence, and Line 7 only decides if Line 8 is asked. For `givenS = false`, this is
`stepIGivenX_parent`. -/
theorem stepIS_parent (givenS : Bool) (H : TargetHyp P ci N t n) (A : Adj)
    (hA : LearnedPast N A) (S : List Vertex) (hS : ∀ s, s ∈ S → s ∈ past N t)
    (hx : (⟨t, n⟩ : Vertex) ∈ S) (p : Vertex) (hp : P p ⟨t + 1, n⟩) :
    p ∈ stepIS ci givenS A ⟨t, n⟩ ⟨t + 1, n⟩ S p := by
  cases givenS with
  | false => exact stepIGivenX_parent H A hA S hx p hp
  | true =>
    by_cases hpS : p ∈ S
    · exact (stepIS_sub ci true A _ _ S p p).mpr (Or.inl hpS)
    · have hpp := H.observed p hp
      have h8 : ci p ⟨t + 1, n⟩ [] = false :=
        H.adj p [] hp (fun _ hs => absurd hs List.not_mem_nil) List.not_mem_nil
      have h9 : ci p ⟨t + 1, n⟩ S = false := H.adj p S hp hS hpS
      have h10 : ci p ⟨t + 1, n⟩ (A p) = false :=
        H.adj p (A p) hp (fun s hs => (hA.past_of_mem hpp hs).1)
          (fun h => (hA.past_of_mem hpp h).2 rfl)
      have e : stepIS ci true A ⟨t, n⟩ ⟨t + 1, n⟩ S p = insertAll S [p] := by
        simp [stepIS, h8, h9, h10]
      rw [e]
      exact (mem_insertAll S [p] p).mpr (Or.inr (List.mem_singleton.mpr rfl))

/-- **Phase I keeps every parent**, for both values of `givenS`, every learned graph `A` with
`LearnedPast` and every candidate order. -/
theorem phaseIS_parents (givenS : Bool) (H : TargetHyp P ci N t n) (A : Adj)
    (hA : LearnedPast N A) :
    ∀ (cands S : List Vertex), (∀ z, z ∈ cands → z ∈ past N t) →
      (∀ s, s ∈ S → s ∈ past N t) → (⟨t, n⟩ : Vertex) ∈ S →
      ∀ p, P p ⟨t + 1, n⟩ → p ∈ S ∨ p ∈ cands →
        p ∈ cands.foldl (stepIS ci givenS A ⟨t, n⟩ ⟨t + 1, n⟩) S
  | [], _, _, _, _, _, _, h => by
    rcases h with h | h
    · exact h
    · exact absurd h List.not_mem_nil
  | z :: cands, S, hc, hS, hx, p, hp, h => by
    rw [List.foldl_cons]
    have hz : z ∈ past N t := hc z (List.mem_cons.mpr (Or.inl rfl))
    have hS' : ∀ s, s ∈ stepIS ci givenS A ⟨t, n⟩ ⟨t + 1, n⟩ S z → s ∈ past N t := by
      intro s hs
      rcases (stepIS_sub ci givenS A _ _ S z s).mp hs with h | ⟨h, _⟩
      · exact hS s h
      · exact h ▸ hz
    apply phaseIS_parents givenS H A hA cands _
      (fun w hw => hc w (List.mem_cons.mpr (Or.inr hw))) hS'
      ((stepIS_sub ci givenS A _ _ S z _).mpr (Or.inl hx)) p hp
    rcases h with h | h
    · exact Or.inl ((stepIS_sub ci givenS A _ _ S z p).mpr (Or.inl h))
    · rcases List.mem_cons.mp h with h | h
      · subst h
        exact Or.inl (stepIS_parent givenS H A hA S hS hx p hp)
      · exact Or.inr h

/-- **Lines 13-15 from any state** with `pa_hat(Y) ⊆ pa(Y) ⊆ S' ⊆ V_{1:t}` and every parent in
`pa_hat(Y)` or in the iteration list: the final `pa_hat(Y)` is `pa(Y)`, for every order. A parent
is kept (`adj`); a non-parent `Z` is removed, because `S' \ {Z}` still contains `pa(Y)`
(`markov`). -/
theorem foldII_correct (H : TargetHyp P ci N t n) :
    ∀ (L : List Vertex) (st : List Vertex × List Vertex),
      (∀ z, z ∈ L → z ∈ past N t) →
      (∀ u, u ∈ st.1 → P u ⟨t + 1, n⟩) →
      (∀ p, P p ⟨t + 1, n⟩ → p ∈ st.2) →
      (∀ s, s ∈ st.2 → s ∈ past N t) →
      (∀ p, P p ⟨t + 1, n⟩ → p ∈ st.1 ∨ p ∈ L) →
      ∀ u, u ∈ (L.foldl (stepII ci ⟨t + 1, n⟩) st).1 ↔ P u ⟨t + 1, n⟩ := by
  intro L
  induction L with
  | nil =>
    intro st _ hP _ _ h5 u
    refine ⟨hP u, ?_⟩
    intro hu
    rcases h5 u hu with h | h
    · exact h
    · exact absurd h List.not_mem_nil
  | cons z L ih =>
    intro st hL hP hS hE h5
    rw [List.foldl_cons]
    have hz : z ∈ past N t := hL z (List.mem_cons.mpr (Or.inl rfl))
    have hfilt : ∀ s, s ∈ st.2.filter (fun s => s != z) ↔ s ∈ st.2 ∧ s ≠ z := by
      intro s
      rw [List.mem_filter]
      simp
    have hL' : ∀ w, w ∈ L → w ∈ past N t := fun w hw => hL w (List.mem_cons.mpr (Or.inr hw))
    by_cases hzp : P z ⟨t + 1, n⟩
    · -- Line 14: a parent is reported dependent and is added
      have hci : ci z ⟨t + 1, n⟩ (st.2.filter (fun s => s != z)) = false :=
        H.adj z _ hzp (fun s hs => hE s ((hfilt s).mp hs).1) (fun h => ((hfilt z).mp h).2 rfl)
      have hstep : stepII ci ⟨t + 1, n⟩ st z = (st.1 ++ [z], st.2) := by
        unfold stepII
        rw [hci]
        rfl
      rw [hstep]
      apply ih (st.1 ++ [z], st.2) hL' ?_ hS hE ?_
      · intro u hu
        rcases List.mem_append.mp hu with hu | hu
        · exact hP u hu
        · rw [List.mem_singleton.mp hu]
          exact hzp
      · intro p hp
        rcases h5 p hp with h | h
        · exact Or.inl (List.mem_append.mpr (Or.inl h))
        · rcases List.mem_cons.mp h with h | h
          · exact Or.inl (List.mem_append.mpr (Or.inr (List.mem_singleton.mpr h)))
          · exact Or.inr h
    · -- Line 15: a non-parent is reported independent given `S' \ {Z} ⊇ pa(Y)`
      have hci : ci z ⟨t + 1, n⟩ (st.2.filter (fun s => s != z)) = true :=
        H.markov z _ hz hzp (fun h => ((hfilt z).mp h).2 rfl)
          (fun s hs => hE s ((hfilt s).mp hs).1)
          (fun p hp => (hfilt p).mpr ⟨hS p hp, fun e => hzp (e ▸ hp)⟩)
      have hstep : stepII ci ⟨t + 1, n⟩ st z = (st.1, st.2.filter (fun s => s != z)) := by
        unfold stepII
        rw [if_pos hci]
      rw [hstep]
      apply ih (st.1, st.2.filter (fun s => s != z)) hL' hP ?_ ?_ ?_
      · intro p hp
        exact (hfilt p).mpr ⟨hS p hp, fun e => hzp (e ▸ hp)⟩
      · intro s hs
        exact hE s ((hfilt s).mp hs).1
      · intro p hp
        rcases h5 p hp with h | h
        · exact Or.inl h
        · rcases List.mem_cons.mp h with h | h
          · exact absurd (h ▸ hp) hzp
          · exact Or.inr h

/-- **One target.** Under `TargetHyp` for `Y = V^n_{t+1}`, Lines 4-15
(`discoverS ci true givenS candOrd ord2 N A t n`) return `pa(Y)`, for `onlyX = true` (Line 11 of
Algorithm 1), both values of `givenS`, every learned graph `A` with `LearnedPast N A` (entries are
earlier observed variables), every candidate order (Line 6) and every order of Line 13. -/
theorem discoverS_correct (givenS : Bool) (H : TargetHyp P ci N t n) (A : Adj)
    (hA : LearnedPast N A) (candOrd ord2 : List Vertex → List Vertex) (hc : IsOrder candOrd)
    (h2 : IsOrder ord2) :
    ∀ u, u ∈ discoverS ci true givenS candOrd ord2 N A t n ↔ P u ⟨t + 1, n⟩ := by
  have hxp : (⟨t, n⟩ : Vertex) ∈ past N t := H.observed _ H.lag1
  have hcand : ∀ z, z ∈ candOrd (candidates N t ⟨t, n⟩) ↔ z ∈ past N t ∧ z ≠ ⟨t, n⟩ := by
    intro z
    rw [hc.mem_iff]
    exact mem_candidates N t _ z
  obtain ⟨S, hSdef⟩ : ∃ S, S = phaseIS ci givenS A ⟨t, n⟩ ⟨t + 1, n⟩
      (candOrd (candidates N t ⟨t, n⟩)) := ⟨_, rfl⟩
  have hSpast : ∀ s, s ∈ S → s ∈ past N t := by
    intro s hs
    rw [hSdef] at hs
    rcases foldIS_mem ci givenS A _ _ _ _ s hs with h | h
    · rw [List.mem_singleton.mp h]
      exact hxp
    · exact ((hcand s).mp h).1
  have hSpa : ∀ p, P p ⟨t + 1, n⟩ → p ∈ S := by
    intro p hp
    rw [hSdef]
    apply phaseIS_parents givenS H A hA _ [⟨t, n⟩] (fun z hz => ((hcand z).mp hz).1)
      (fun s hs => by rw [List.mem_singleton.mp hs]; exact hxp) (List.mem_singleton.mpr rfl) p hp
    by_cases hpx : p = ⟨t, n⟩
    · exact Or.inl (List.mem_singleton.mpr hpx)
    · exact Or.inr ((hcand p).mpr ⟨H.observed p hp, hpx⟩)
  have hL : ∀ z, z ∈ ord2 (candsII true ⟨t, n⟩ S) ↔ z ∈ S ∧ z ≠ ⟨t, n⟩ := by
    intro z
    rw [h2.mem_iff]
    unfold candsII initPa
    simp
  unfold discoverS
  rw [← hSdef]
  unfold phaseII
  apply foldII_correct H
  · intro z hz
    exact hSpast z ((hL z).mp hz).1
  · intro u hu
    simp [initPa] at hu
    rw [hu]
    exact H.lag1
  · exact hSpa
  · exact hSpast
  · intro p hp
    by_cases hpx : p = ⟨t, n⟩
    · left
      simp [initPa, hpx]
    · exact Or.inr ((hL p).mpr ⟨hSpa p hp, hpx⟩)

end OneTarget

/-! ## Exactness for the whole run -/

/-- **Exactness of Algorithm 1** (`onlyX = true`, Line 11 of Algorithm 1; both values of `givenS`).
For every target `Y = V^n_{t+1}` (`1 ≤ t < T`) that satisfies `TargetHyp`, the returned entry `A[Y]`
is `pa(Y)`. No hypothesis is made on other targets, on the first step or on the learned graph. Every
candidate order (Line 6), Phase II order (Line 13) and series order (Line 3). -/
theorem itpdS_correct (P : Vertex → Vertex → Prop) (ci : CI) (givenS : Bool)
    (candOrd ord2 : List Vertex → List Vertex) (hc : IsOrder candOrd) (h2 : IsOrder ord2)
    (N T : Nat) (serOrd : Nat → List Nat) (hser : IsSeriesOrder N serOrd) (t n : Nat)
    (htT : t < T) (H : TargetHyp P ci N t n) :
    ∀ u, u ∈ itpdS ci true givenS candOrd ord2 N T serOrd ⟨t + 1, n⟩ ↔ P u ⟨t + 1, n⟩ := by
  have hxp := mem_past.mp (H.observed _ H.lag1)
  rcases foldTime_last _ serOrd t n (LearnedPast N) (timeSteps T) emptyAdj
      (fun B s m hs hm hB => runSeriesS_learnedPast ci true givenS candOrd ord2 hc h2 N B s m
        (mem_timeSteps.mp hs).1 ((mem_serOrd hser s m).mp hm) hB)
      (learnedPast_empty N) (mem_timeSteps.mpr ⟨hxp.1, htT⟩)
      ((mem_serOrd hser t n).mpr ⟨hxp.2.2.1, hxp.2.2.2⟩) with ⟨A', hA', he⟩
  unfold itpdS run
  rw [he]
  exact discoverS_correct givenS H A' hA' candOrd ord2 hc h2

end ITPD
