import ITPD.Graph

/-!
# Separation.lean : the separation relation that models the CI answers

`MSep.msep E z y S`: `z` and `y` are separated by `S` in the graph `E` iff `z ≠ y` and no walk
joins them in the moral graph of the subgraph induced by the ancestral closure of `{z, y} ∪ S`,
where a walk may not visit a member of `S` other than `z` and `y` (the moral-ancestral criterion
of Lauritzen, Dawid, Larsen and Leimer, 1990). The equivalence of this criterion with path-based
d-separation is their theorem; it is not formalized here.

The second part proves the two facts about this relation that the exactness proof needs, for
every graph whose edges point forward in time: an edge is never separated, and a target is
separated from an earlier non-parent by every earlier set that contains the parents of the
target.
-/

set_option autoImplicit false

namespace ITPD

namespace MSep

/-- `v` is in the ancestral closure of `{z, y} ∪ S`: it is one of these vertices or an ancestor
of one of them. -/
def InAn (E : Vertex → Vertex → Prop) (z y : Vertex) (S : List Vertex) (v : Vertex) : Prop :=
  ∃ a, (a = z ∨ a = y ∨ a ∈ S) ∧ (v = a ∨ Anc E v a)

/-- Moral graph of the subgraph induced by `W`: `u` and `v` are joined by an edge of `E`, or are
parents of a common child, everything inside `W`. -/
def Moral (E : Vertex → Vertex → Prop) (W : Vertex → Prop) (u v : Vertex) : Prop :=
  W u ∧ W v ∧ (E u v ∨ E v u ∨ ∃ c, W c ∧ E u c ∧ E v c)

/-- One step of a walk in the moral graph of `W` whose two ends are not blocked by `B`. -/
def Step (E : Vertex → Vertex → Prop) (W B : Vertex → Prop) (u v : Vertex) : Prop :=
  Moral E W u v ∧ ¬ B u ∧ ¬ B v

/-- Walks (possibly of length zero) of a step relation. -/
inductive Walk (R : Vertex → Vertex → Prop) : Vertex → Vertex → Prop
  | refl (v : Vertex) : Walk R v v
  | tail {u w v : Vertex} : Walk R u w → R w v → Walk R u v

/-- The blocked vertices: the members of `S` other than the two endpoints. -/
def Blocked (z y : Vertex) (S : List Vertex) (v : Vertex) : Prop := v ∈ S ∧ v ≠ z ∧ v ≠ y

/-- **Moral-ancestral separation** of `z` and `y` by `S`. -/
def msep (E : Vertex → Vertex → Prop) (z y : Vertex) (S : List Vertex) : Prop :=
  z ≠ y ∧ ¬ Walk (Step E (InAn E z y S) (Blocked z y S)) z y

variable {E : Vertex → Vertex → Prop}

/-- The moral graph is symmetric. -/
theorem moral_symm {W : Vertex → Prop} {u v : Vertex} (h : Moral E W u v) : Moral E W v u := by
  rcases h with ⟨hu, hv, h | h | ⟨c, hc, h1, h2⟩⟩
  · exact ⟨hv, hu, Or.inr (Or.inl h)⟩
  · exact ⟨hv, hu, Or.inl h⟩
  · exact ⟨hv, hu, Or.inr (Or.inr ⟨c, hc, h2, h1⟩)⟩

/-- Last step of a walk. -/
theorem Walk.cases_tail {R : Vertex → Vertex → Prop} {u v : Vertex} (h : Walk R u v) :
    u = v ∨ ∃ w, Walk R u w ∧ R w v := by
  cases h with
  | refl => exact Or.inl rfl
  | tail h1 h2 => exact Or.inr ⟨_, h1, h2⟩

end MSep

/-! ## Separation in a graph with forward edges -/

section Forward

open MSep

variable {E : Vertex → Vertex → Prop}

/-- A direct edge is never separated, for every conditioning set and every edge relation. -/
theorem not_msep_of_edge {z y : Vertex} (S : List Vertex) (he : E z y) : ¬ msep E z y S := by
  intro hm
  exact hm.2 (Walk.tail (Walk.refl _)
    ⟨⟨⟨z, Or.inl rfl, Or.inl rfl⟩, ⟨y, Or.inr (Or.inl rfl), Or.inl rfl⟩, Or.inl he⟩,
      fun hb => hb.2.1 rfl, fun hb => hb.2.2 rfl⟩)

/-- In a graph with forward edges, the ancestral closure of `{z, y} ∪ S` with `z` and `S` earlier
than `y` has no vertex later than `y`, and only `y` at the time of `y`. -/
theorem inAn_time_le_of_forward (hf : ∀ u v, E u v → u.time < v.time) {z y : Vertex}
    {S : List Vertex} (hz : z.time < y.time) (hS : ∀ s, s ∈ S → s.time < y.time) {v : Vertex}
    (hv : InAn E z y S v) : v = y ∨ v.time < y.time := by
  rcases hv with ⟨a, ha, hva⟩
  have hat : a = y ∨ a.time < y.time := by
    rcases ha with ha | ha | ha
    · subst ha
      exact Or.inr hz
    · exact Or.inl ha
    · exact Or.inr (hS _ ha)
  rcases hva with hva | hva
  · subst hva
    exact hat
  · have h1 := Anc.time_lt hf hva
    rcases hat with hat | hat
    · subst hat
      exact Or.inr h1
    · exact Or.inr (Nat.lt_trans h1 hat)

/-- Every moral neighbor of `y` in that closure is a parent of `y` (`y` has no child there). -/
theorem moral_nbr_of_target_of_forward (hf : ∀ u v, E u v → u.time < v.time) {z y : Vertex}
    {S : List Vertex} (hz : z.time < y.time) (hS : ∀ s, s ∈ S → s.time < y.time) {v : Vertex}
    (hm : Moral E (InAn E z y S) y v) : E v y := by
  rcases hm with ⟨_, hWv, he | he | ⟨c, hWc, h1, _⟩⟩
  · exfalso
    have h2 := hf _ _ he
    rcases inAn_time_le_of_forward hf hz hS hWv with e | h3
    · rw [e] at h2
      exact Nat.lt_irrefl _ h2
    · exact Nat.lt_irrefl _ (Nat.lt_trans h2 h3)
  · exact he
  · exfalso
    have h2 := hf _ _ h1
    rcases inAn_time_le_of_forward hf hz hS hWc with e | h3
    · rw [e] at h2
      exact Nat.lt_irrefl _ h2
    · exact Nat.lt_irrefl _ (Nat.lt_trans h2 h3)

/-- **Local Markov property of `y` with weak union, any forward graph.** If `z` and the members
of `S` are earlier than `y`, every parent of `y` is in `S` and `z` is not a parent of `y`, then
`S` separates `z` and `y` (a member of `S` equal to `z` is ignored, see `MSep.Blocked`). The
parents of `y` may have latent ancestors; only the parents of `y` themselves must be in `S`. -/
theorem msep_local_markov (hf : ∀ u v, E u v → u.time < v.time) {z y : Vertex}
    {S : List Vertex} (hz : z.time < y.time) (hzp : ¬ E z y)
    (hS : ∀ s, s ∈ S → s.time < y.time) (hpa : ∀ p, E p y → p ∈ S) : msep E z y S := by
  refine ⟨?_, ?_⟩
  · intro e
    rw [e] at hz
    exact Nat.lt_irrefl _ hz
  · intro hw
    rcases Walk.cases_tail hw with e | ⟨w, _, hs⟩
    · rw [e] at hz
      exact Nat.lt_irrefl _ hz
    · -- the last step enters `y` from a parent `w` of `y`; `w ∈ S`, so `w` is blocked
      have hwy : E w y := moral_nbr_of_target_of_forward hf hz hS (moral_symm hs.1)
      have hwS : w ∈ S := hpa w hwy
      by_cases hwz : w = z
      · rw [hwz] at hwy
        exact hzp hwy
      · refine hs.2.1 ⟨hwS, hwz, ?_⟩
        intro e
        have h2 := hf _ _ hwy
        rw [e] at h2
        exact Nat.lt_irrefl _ h2

end Forward

end ITPD
