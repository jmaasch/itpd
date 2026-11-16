/-!
# Graph.lean : vertices, ancestors and the observed past

A vertex is a pair (time, series): the variable `V^n_t` of the paper is `⟨t, n⟩`. A graph is a
directed edge relation `E u v` ("`u` is a parent of `v`"). `past N t` lists the observed
variables `V_{1:t}` of `N` series.

Only Lean's core library is used.
-/

set_option autoImplicit false

namespace ITPD

/-- A vertex `V^n_t`: `time = t`, `series = n`. The paper counts both from 1. -/
structure Vertex where
  time : Nat
  series : Nat
deriving DecidableEq, Repr

/-! ## Ancestors for an arbitrary edge relation -/

/-- `Anc E u v`: there is a directed path with at least one edge from `u` to `v`
(`u` is a strict ancestor of `v`). -/
inductive Anc (E : Vertex → Vertex → Prop) : Vertex → Vertex → Prop
  | edge {u v : Vertex} : E u v → Anc E u v
  | step {u w v : Vertex} : Anc E u w → E w v → Anc E u v

section

variable {E : Vertex → Vertex → Prop}

/-- Ancestors are strictly earlier, for every relation whose edges point forward in time. -/
theorem Anc.time_lt (hf : ∀ u v, E u v → u.time < v.time) {u v : Vertex} (h : Anc E u v) :
    u.time < v.time := by
  induction h with
  | edge h => exact hf _ _ h
  | step _ h ih => exact Nat.lt_trans ih (hf _ _ h)

end

/-! ## Enumerations of the vertices -/

/-- The `N` vertices of time step `t`, in increasing series index. -/
def slice (N t : Nat) : List Vertex := (List.range N).map (fun i => ⟨t, i + 1⟩)

/-- The vertices of the first `t` time steps (`V_{1:t}`), in ascending time, then series. -/
def past (N t : Nat) : List Vertex := (List.range t).flatMap (fun s => slice N (s + 1))

/-- Members of a time step. -/
theorem mem_slice {N t : Nat} {v : Vertex} :
    v ∈ slice N t ↔ v.time = t ∧ 1 ≤ v.series ∧ v.series ≤ N := by
  unfold slice
  rw [List.mem_map]
  constructor
  · rintro ⟨i, hi, rfl⟩
    have := List.mem_range.mp hi
    exact ⟨rfl, by simp, by simp; omega⟩
  · rintro ⟨h1, h2, h3⟩
    refine ⟨v.series - 1, List.mem_range.mpr (by omega), ?_⟩
    cases v with
    | mk vt vs =>
      simp at h1 h2 h3 ⊢
      omega

/-- Members of `V_{1:t}`. -/
theorem mem_past {N t : Nat} {v : Vertex} :
    v ∈ past N t ↔ 1 ≤ v.time ∧ v.time ≤ t ∧ 1 ≤ v.series ∧ v.series ≤ N := by
  unfold past
  rw [List.mem_flatMap]
  constructor
  · rintro ⟨s, hs, hv⟩
    have := List.mem_range.mp hs
    have := mem_slice.mp hv
    omega
  · rintro ⟨h1, h2, h3, h4⟩
    exact ⟨v.time - 1, List.mem_range.mpr (by omega), mem_slice.mpr ⟨by omega, h3, h4⟩⟩

end ITPD
