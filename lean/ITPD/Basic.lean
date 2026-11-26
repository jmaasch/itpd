import ITPD.Algorithm

/-!
# Basic.lean : list facts, the orders of Lines 6 and 13, and the loops of Lines 2-3

* Membership lemmas for `insertAll`, the candidate list of Line 4, the time steps of Line 2 and
  the series of Line 3.
* `descTime` (Line 6) and `ascNode` (Line 13) permute their input (`isOrder_descTime`,
  `isOrder_ascNode`). `descTime` sorts in descending time, `ascNode` in ascending time, and both in
  increasing series index inside a time step (`descTime_sorted`, `ascNode_sorted`). Algorithm 1
  states the order in time only.
* Lemmas about the loops of Lines 2-3 and the update of Line 16. They hold for `run d` with every
  function `d` as the loop body; `itpdS` uses them with `d` equal to Lines 4-15 (`discoverS`).
-/

set_option autoImplicit false

namespace ITPD

/-! ## Membership lemmas -/

/-- Members of `S ∪ L`. -/
theorem mem_insertAll (S L : List Vertex) (v : Vertex) :
    v ∈ insertAll S L ↔ v ∈ S ∨ v ∈ L := by
  unfold insertAll
  induction L generalizing S with
  | nil => simp
  | cons a L ih =>
    rw [List.foldl_cons, ih, List.mem_insert_iff, List.mem_cons]
    constructor
    · rintro ((h | h) | h)
      · exact Or.inr (Or.inl h)
      · exact Or.inl h
      · exact Or.inr (Or.inr h)
    · rintro (h | h | h)
      · exact Or.inl (Or.inr h)
      · exact Or.inl (Or.inl h)
      · exact Or.inr h

/-- An order keeps the members. -/
theorem IsOrder.mem_iff {f : List Vertex → List Vertex} (hf : IsOrder f) (L : List Vertex)
    (v : Vertex) : v ∈ f L ↔ v ∈ L := by
  exact (hf L).mem_iff

/-- Members of the candidate list of Line 4. -/
theorem mem_candidates (N t : Nat) (x z : Vertex) :
    z ∈ candidates N t x ↔ z ∈ past N t ∧ z ≠ x := by
  unfold candidates
  rw [List.mem_filter]
  simp

/-- Members of `1, …, N`. -/
theorem mem_seriesList {N n : Nat} : n ∈ seriesList N ↔ 1 ≤ n ∧ n ≤ N := by
  unfold seriesList
  rw [List.mem_map]
  constructor
  · rintro ⟨i, hi, rfl⟩
    have := List.mem_range.mp hi
    omega
  · rintro ⟨h1, h2⟩
    exact ⟨n - 1, List.mem_range.mpr (by omega), by omega⟩

/-- The time steps of Line 2 are `1 ≤ t < T`. -/
theorem mem_timeSteps {T t : Nat} : t ∈ timeSteps T ↔ 1 ≤ t ∧ t < T := by
  unfold timeSteps
  rw [List.mem_map]
  constructor
  · rintro ⟨i, hi, rfl⟩
    have := List.mem_range.mp hi
    omega
  · rintro ⟨h1, h2⟩
    exact ⟨t - 1, List.mem_range.mpr (by omega), by omega⟩

/-- A series order visits the series `1, …, N`. -/
theorem mem_serOrd {N : Nat} {serOrd : Nat → List Nat} (h : IsSeriesOrder N serOrd) (t n : Nat) :
    n ∈ serOrd t ↔ 1 ≤ n ∧ n ≤ N :=
  ((h t).mem_iff).trans mem_seriesList

/-! ## The orders of Lines 3, 6 and 13 -/

theorem isOrder_descTime : IsOrder descTime := fun L => List.mergeSort_perm L _

theorem isOrder_ascNode : IsOrder ascNode := fun L => List.mergeSort_perm L _

theorem isSeriesOrder_natSeries (N : Nat) : IsSeriesOrder N (natSeries N) :=
  fun _ => List.Perm.refl _

/-- Line 6: `descTime` returns its input in descending time, and in increasing series index
inside a time step. -/
theorem descTime_sorted (L : List Vertex) :
    (descTime L).Pairwise
      (fun a b => b.time < a.time ∨ (a.time = b.time ∧ a.series ≤ b.series)) := by
  have h := List.pairwise_mergeSort
    (le := fun a b : Vertex => decide (b.time < a.time) ||
      (decide (a.time = b.time) && decide (a.series ≤ b.series)))
    (fun a b c hab hbc => by
      simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hab hbc ⊢
      omega)
    (fun a b => by
      simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq]
      omega) L
  exact h.imp (fun {a b} hab => by
    simpa only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] using hab)

/-- Line 13: `ascNode` returns its input in ascending time, and in increasing series index inside
a time step. -/
theorem ascNode_sorted (L : List Vertex) :
    (ascNode L).Pairwise
      (fun a b => a.time < b.time ∨ (a.time = b.time ∧ a.series ≤ b.series)) := by
  have h := List.pairwise_mergeSort
    (le := fun a b : Vertex => decide (a.time < b.time) ||
      (decide (a.time = b.time) && decide (a.series ≤ b.series)))
    (fun a b c hab hbc => by
      simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hab hbc ⊢
      omega)
    (fun a b => by
      simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq]
      omega) L
  exact h.imp (fun {a b} hab => by
    simpa only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] using hab)

/-! ## Loop lemmas for an arbitrary body `d` -/

section Loops

variable (d : Adj → Nat → Nat → List Vertex)

/-- The loop of Line 3 at step `t` changes only entries of step `t + 1` of the visited series. -/
theorem foldSeries_frame (t : Nat) :
    ∀ (L : List Nat) (A : Adj) (v : Vertex), (v.time ≠ t + 1 ∨ v.series ∉ L) →
      (L.foldl (runSeries d t) A) v = A v
  | [], _, _, _ => rfl
  | m :: L, A, v, hv => by
    rw [List.foldl_cons]
    have hv' : v.time ≠ t + 1 ∨ v.series ∉ L := by
      rcases hv with hv | hv
      · exact Or.inl hv
      · exact Or.inr (fun h => hv (List.mem_cons.mpr (Or.inr h)))
    rw [foldSeries_frame t L _ v hv']
    have hne : v ≠ (⟨t + 1, m⟩ : Vertex) := by
      intro e
      subst e
      rcases hv with hv | hv
      · exact hv rfl
      · exact hv (List.mem_cons.mpr (Or.inl rfl))
    simp [runSeries, update, hne]

/-- The loop of Line 2 over the steps `Ts` changes only entries of steps `s + 1`, `s ∈ Ts`, of
the series visited at step `s`. -/
theorem foldTime_frame (serOrd : Nat → List Nat) :
    ∀ (Ts : List Nat) (A : Adj) (v : Vertex),
      (∀ s, s ∈ Ts → v.time ≠ s + 1 ∨ v.series ∉ serOrd s) →
      (Ts.foldl (runTime d serOrd) A) v = A v
  | [], _, _, _ => rfl
  | s :: Ts, A, v, hv => by
    rw [List.foldl_cons,
      foldTime_frame serOrd Ts _ v (fun s' hs' => hv s' (List.mem_cons.mpr (Or.inr hs')))]
    exact foldSeries_frame d s (serOrd s) A v (hv s (List.mem_cons.mpr (Or.inl rfl)))

/-- A property `Q` of the adjacency list that every visited update keeps holds after the loop of
Line 3. -/
theorem foldSeries_keep (t : Nat) (Q : Adj → Prop) :
    ∀ (L : List Nat) (A : Adj), (∀ B m, m ∈ L → Q B → Q (runSeries d t B m)) → Q A →
      Q (L.foldl (runSeries d t) A)
  | [], _, _, hA => hA
  | m :: L, A, hQ, hA => by
    rw [List.foldl_cons]
    exact foldSeries_keep t Q L _ (fun B m' hm' hB => hQ B m' (List.mem_cons.mpr (Or.inr hm')) hB)
      (hQ A m (List.mem_cons.mpr (Or.inl rfl)) hA)

/-- **The final entry of `Y = V^n_{t+1}` after the loop of Line 3** is the body evaluated on an
intermediate adjacency list `A'` that has the property `Q` and agrees with the start list `A`
outside step `t + 1`. -/
theorem foldSeries_last (t n : Nat) (Q : Adj → Prop) :
    ∀ (L : List Nat) (A : Adj), (∀ B m, m ∈ L → Q B → Q (runSeries d t B m)) → Q A → n ∈ L →
      ∃ A' : Adj, Q A' ∧ (∀ v : Vertex, v.time ≠ t + 1 → A' v = A v) ∧
        (L.foldl (runSeries d t) A) ⟨t + 1, n⟩ = d A' t n
  | [], _, _, _, hn => absurd hn List.not_mem_nil
  | m :: L, A, hQ, hA, hn => by
    rw [List.foldl_cons]
    have hQL : ∀ B m', m' ∈ L → Q B → Q (runSeries d t B m') :=
      fun B m' hm' hB => hQ B m' (List.mem_cons.mpr (Or.inr hm')) hB
    have hA1 : Q (runSeries d t A m) := hQ A m (List.mem_cons.mpr (Or.inl rfl)) hA
    have hagree : ∀ v : Vertex, v.time ≠ t + 1 → runSeries d t A m v = A v := by
      intro v hv
      have hne : v ≠ (⟨t + 1, m⟩ : Vertex) := by
        intro e
        subst e
        exact hv rfl
      simp [runSeries, update, hne]
    by_cases hnL : n ∈ L
    · rcases foldSeries_last t n Q L _ hQL hA1 hnL with ⟨A', hQ', hag, he⟩
      exact ⟨A', hQ', fun v hv => (hag v hv).trans (hagree v hv), he⟩
    · have hnm : n = m := by
        rcases List.mem_cons.mp hn with h | h
        · exact h
        · exact absurd h hnL
      subst hnm
      refine ⟨A, hA, fun _ _ => rfl, ?_⟩
      rw [foldSeries_frame d t L _ ⟨t + 1, n⟩ (Or.inr hnL)]
      simp [runSeries, update]

/-- **The final entry of `Y = V^n_{t+1}` after the loop of Line 2** is the body evaluated on an
intermediate adjacency list that has the property `Q` (kept by every visited update). -/
theorem foldTime_last (serOrd : Nat → List Nat) (t n : Nat) (Q : Adj → Prop) :
    ∀ (Ts : List Nat) (A : Adj),
      (∀ B s m, s ∈ Ts → m ∈ serOrd s → Q B → Q (runSeries d s B m)) → Q A →
      t ∈ Ts → n ∈ serOrd t →
      ∃ A' : Adj, Q A' ∧ (Ts.foldl (runTime d serOrd) A) ⟨t + 1, n⟩ = d A' t n
  | [], _, _, _, ht, _ => absurd ht List.not_mem_nil
  | s :: Ts, A, hQ, hA, ht, hn => by
    rw [List.foldl_cons]
    have hQs : ∀ B m, m ∈ serOrd s → Q B → Q (runSeries d s B m) :=
      fun B m hm hB => hQ B s m (List.mem_cons.mpr (Or.inl rfl)) hm hB
    have hQT : ∀ B s' m, s' ∈ Ts → m ∈ serOrd s' → Q B → Q (runSeries d s' B m) :=
      fun B s' m hs' hm hB => hQ B s' m (List.mem_cons.mpr (Or.inr hs')) hm hB
    have hA1 : Q (runTime d serOrd A s) := foldSeries_keep d s Q (serOrd s) A hQs hA
    by_cases htT : t ∈ Ts
    · exact foldTime_last serOrd t n Q Ts _ hQT hA1 htT hn
    · have hts : t = s := by
        rcases List.mem_cons.mp ht with h | h
        · exact h
        · exact absurd h htT
      subst hts
      rw [foldTime_frame d serOrd Ts _ ⟨t + 1, n⟩ (by
        intro s' hs'
        left
        intro e
        have : t = s' := by simp at e; omega
        exact htT (this ▸ hs'))]
      rcases foldSeries_last d t n Q (serOrd t) A hQs hA hn with ⟨A', hQ', _, he⟩
      exact ⟨A', hQ', he⟩

end Loops

end ITPD
