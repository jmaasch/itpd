import ITPD.Graph

/-!
# Algorithm.lean : Algorithm 1 (ITPD) as Lean functions (definitions only)

"Line k" means Line k of Algorithm 1 of the paper:

* Line 1: `A ← {V^i_1 ↦ ∅}`. Lines 2-3: loops over `t` and `n`.
* Line 4: `X ← V^n_t`, `Y ← V^n_{t+1}`, `Z ← V_{1:t} \ {X, Y}`.
* Line 5: `S ← {X}`. Line 6: for `Z` in descending time order.
* Line 7: if `Z ∉ an_hat(S)` then Line 8: if `CI(Z ⊥ Y)` then continue.
* Line 9: if `CI(Z ⊥ Y | S)` then continue.
* Line 10: if `CI(Z ⊥̸ Y | pa_hat(Z))` then `S ← S ∪ {Z}`.
* Line 11: `pa_hat(Y) ← {X}`. Line 12: `S' ← S`, `Z' ← S \ pa_hat(Y)`.
* Line 13: for `Z` in `Z'` in ascending time order.
* Line 14: if `CI(Z ⊥̸ Y | S' \ {Z})` then `pa_hat(Y) ← pa_hat(Y) ∪ {Z}`.
* Line 15: else `S' ← S' \ {Z}`.
* Line 16: `A[Y] ← pa_hat(Y)`.

Sets are lists. All functions are total. The parameters are the CI answers `ci`, the learned
adjacency list `A`, the candidate order `candOrd` (Line 6), the Phase II order `ord2` (Line 13)
and the series order `serOrd` (Line 3).

`itpdS` has two Boolean parameters, `givenS` and `onlyX`; Algorithm 1 is the instance with both set
to `true`.
* `givenS` sets the conditioning set of the Line 9 test. `true`: `S` (Line 9 of Algorithm 1).
  `false`: `{X}`, and that test counts only when `Z` and `X` are reported dependent given the empty
  set.
* `onlyX` sets the start of `pa_hat(Y)` in Line 11. `true`: `{X}` (Line 11 of Algorithm 1). `false`:
  `{X} ∪ (S ∩ V_t)`, where `V_t` is the time step of `X`.
* `itpdPaper` in `Paper.lean` sets both to `true` and uses the orders `descTime`, `ascNode` and
  `natSeries N`.

The significance levels `α_1` and `α_2` of Algorithm 1 are not modeled: a CI test is its Boolean
answer.
-/

set_option autoImplicit false

namespace ITPD

/-- The answers of the CI test: `ci z y S = true` iff the test reports `Z ⊥ Y | S`. The argument
order is the same everywhere: candidate, target, conditioning set. -/
abbrev CI := Vertex → Vertex → List Vertex → Bool

/-- The adjacency list `A`: `A v` is the learned parent set `pa_hat(v)`. -/
abbrev Adj := Vertex → List Vertex

/-! ## Quantities read from the learned graph -/

/-- Ancestors of `v` in the learned graph along paths with at most `k` edges. -/
def ancFuel (A : Adj) : Nat → Vertex → List Vertex
  | 0, _ => []
  | k + 1, v => A v ++ (A v).flatMap (ancFuel A k)

/-- `an_hat(v)`: the strict ancestors of `v` in the learned graph. A path into `v` in a graph
with forward-in-time edges has at most `v.time` edges, so `v.time` is enough fuel. -/
def anHat (A : Adj) (v : Vertex) : List Vertex := ancFuel A v.time v

/-- Line 7, `Z ∈ an_hat(S)`: `z` is a learned strict ancestor of a member of `S`. -/
def inAnHatOfSet (A : Adj) (S : List Vertex) (z : Vertex) : Bool :=
  S.any (fun s => decide (z ∈ anHat A s))

/-- Result `S ∪ L`, adding the members of `L` one at a time. A member already in the list is not
added again, so no duplicate is created. -/
def insertAll (S L : List Vertex) : List Vertex := L.foldl (fun acc v => acc.insert v) S

/-! ## Phase I (Lines 5-10) -/

/-- Line 4: `Z = V_{1:t} \ {X, Y}` in ascending time, then series, before the candidate order of
Line 6 is applied. `Y` is at time `t + 1` and is not in `V_{1:t}`. -/
def candidates (N t : Nat) (x : Vertex) : List Vertex := (past N t).filter (fun z => z != x)

/-- Lines 7-10 for `givenS = false` (called by `stepIS`). Its second test is `Z ⊥ Y | {X}`
(`ci z y [x]`), and counts only when `Z` and `X` are reported dependent given the empty set
(`ci z x [] = false`). Lines 7, 8 and 10 are the same as for `givenS = true`. -/
def stepIGivenX (ci : CI) (A : Adj) (x y : Vertex) (S : List Vertex) (z : Vertex) :
    List Vertex :=
  if (!(inAnHatOfSet A S z) && ci z y []) = true then S   -- Lines 7-8
  else if (!(ci z x []) && ci z y [x]) = true then S      -- Line 9 for `givenS = false`
  else if ci z y (A z) = false then insertAll S [z]       -- Line 10
  else S

/-- Lines 7-10 for one candidate `z` and the current set `S`; returns `S` after the step.
`givenS = true`: Line 9 of Algorithm 1, the test given `S` as it is when `z` is visited.
`givenS = false`: `stepIGivenX`. The argument `x` is used for `givenS = false` only. -/
def stepIS (ci : CI) : Bool → Adj → Vertex → Vertex → List Vertex → Vertex → List Vertex
  | false, A, x, y, S, z => stepIGivenX ci A x y S z
  | true, A, _, y, S, z =>
    if (!(inAnHatOfSet A S z) && ci z y []) = true then S   -- Lines 7-8: `Z ∉ an_hat(S)`, `Z ⊥ Y`
    else if ci z y S = true then S                          -- Line 9: `Z ⊥ Y | S`
    else if ci z y (A z) = false then insertAll S [z]       -- Line 10: `Z ⊥̸ Y | pa_hat(Z)`
    else S

/-- Phase I (Lines 5-10): `S ← {X}` (Line 5), then the loop of Line 6 over `cands`. -/
def phaseIS (ci : CI) (givenS : Bool) (A : Adj) (x y : Vertex) (cands : List Vertex) :
    List Vertex :=
  cands.foldl (stepIS ci givenS A x y) [x]

/-! ## Phase II (Lines 11-15) -/

/-- `{X} ∪ (S ∩ V_t)`, where `V_t` is the time step of `X`. Start of `pa_hat(Y)` for
`onlyX = false`, used by `initPa`. -/
def promote (x : Vertex) (S : List Vertex) : List Vertex :=
  x :: S.filter (fun z => z.time == x.time && z != x)

/-- Start of `pa_hat(Y)` in Line 11. `onlyX = true`: `{X}`, which is Line 11 of Algorithm 1
(`pa_hat(Y) ← {X}`). `onlyX = false`: `promote x S`. -/
def initPa (onlyX : Bool) (x : Vertex) (S : List Vertex) : List Vertex :=
  if onlyX then [x] else promote x S

/-- Line 12: `Z' = S \ pa_hat(Y)`, with `pa_hat(Y)` from Line 11. -/
def candsII (onlyX : Bool) (x : Vertex) (S : List Vertex) : List Vertex :=
  S.filter (fun z => decide (z ∉ initPa onlyX x S))

/-- Lines 14-15 for one `z`, on the state (`pa_hat(Y)`, `S'`). -/
def stepII (ci : CI) (y : Vertex) (st : List Vertex × List Vertex) (z : Vertex) :
    List Vertex × List Vertex :=
  if ci z y (st.2.filter (fun s => s != z)) = true
  then (st.1, st.2.filter (fun s => s != z))     -- Line 15: independent, S' ← S' \ {Z}
  else (st.1 ++ [z], st.2)                       -- Line 14: dependent, pa_hat(Y) ← pa_hat(Y) ∪ {Z}

/-- Phase II (Lines 11-15): start state (`pa_hat(Y)` of Line 11, `S' = S`), loop of Line 13 over
`Z'` in the order `ord2`, body `stepII`. Returns `pa_hat(Y)`. -/
def phaseII (ci : CI) (ord2 : List Vertex → List Vertex) (onlyX : Bool) (x y : Vertex)
    (S : List Vertex) : List Vertex :=
  ((ord2 (candsII onlyX x S)).foldl (stepII ci y) (initPa onlyX x S, S)).1

/-! ## One target (Lines 4-15) and the loops (Lines 1-3 and 16) -/

/-- Lines 4-15 for the target `Y = V^n_{t+1}`, `X = V^n_t`, with the learned graph `A`. -/
def discoverS (ci : CI) (onlyX givenS : Bool) (candOrd ord2 : List Vertex → List Vertex)
    (N : Nat) (A : Adj) (t n : Nat) : List Vertex :=
  phaseII ci ord2 onlyX ⟨t, n⟩ ⟨t + 1, n⟩
    (phaseIS ci givenS A ⟨t, n⟩ ⟨t + 1, n⟩ (candOrd (candidates N t ⟨t, n⟩)))

/-- Line 16: `A[Y] ← pa_hat(Y)`. -/
def update (A : Adj) (y : Vertex) (ps : List Vertex) : Adj := fun v => if v = y then ps else A v

/-- Line 16 for series `n` at step `t`, with the body `d A t n` (Lines 4-15). -/
def runSeries (d : Adj → Nat → Nat → List Vertex) (t : Nat) (A : Adj) (n : Nat) : Adj :=
  update A ⟨t + 1, n⟩ (d A t n)

/-- The loop of Line 3 at step `t`, series in the order `serOrd t`. -/
def runTime (d : Adj → Nat → Nat → List Vertex) (serOrd : Nat → List Nat) (A : Adj) (t : Nat) :
    Adj :=
  (serOrd t).foldl (runSeries d t) A

/-- The time steps `1, …, T - 1` of the loop of Line 2. -/
def timeSteps (T : Nat) : List Nat := (List.range (T - 1)).map (fun k => k + 1)

/-- Line 1: every entry of `A` is empty at the start (in particular those of the first step). -/
def emptyAdj : Adj := fun _ => []

/-- Lines 1-3 and 16: start from the empty lists (Line 1), loop over `t = 1, …, T - 1`. -/
def run (d : Adj → Nat → Nat → List Vertex) (T : Nat) (serOrd : Nat → List Nat) : Adj :=
  (timeSteps T).foldl (runTime d serOrd) emptyAdj

/-- **Algorithm 1 with two Boolean parameters**, `onlyX` (Line 11) and `givenS` (Line 9), candidate
order `candOrd` (Line 6), Phase II order `ord2` (Line 13) and series order `serOrd` (Line 3).
Returns the adjacency list `A`. Algorithm 1 of the paper is `itpdPaper` (`Paper.lean`):
`itpdS ci true true descTime ascNode N T (natSeries N)`. -/
def itpdS (ci : CI) (onlyX givenS : Bool) (candOrd ord2 : List Vertex → List Vertex)
    (N T : Nat) (serOrd : Nat → List Nat) : Adj :=
  run (fun A t n => discoverS ci onlyX givenS candOrd ord2 N A t n) T serOrd

/-! ## Orders -/

/-- An order on vertex lists: any function that permutes its input. -/
def IsOrder (f : List Vertex → List Vertex) : Prop := ∀ L, (f L).Perm L

/-- The series `1, …, N` in increasing order. -/
def seriesList (N : Nat) : List Nat := (List.range N).map (fun i => i + 1)

/-- A series order: at every step the series `1, …, N` are visited in some order, each once. -/
def IsSeriesOrder (N : Nat) (serOrd : Nat → List Nat) : Prop :=
  ∀ t, (serOrd t).Perm (seriesList N)

/-- The series order of Line 3: `1, …, N` at every step. -/
def natSeries (N : Nat) : Nat → List Nat := fun _ => seriesList N

/-- The candidate order of Line 6: descending time, then increasing series index. -/
def descTime (L : List Vertex) : List Vertex :=
  L.mergeSort (fun a b => decide (b.time < a.time) ||
    (decide (a.time = b.time) && decide (a.series ≤ b.series)))

/-- The Phase II order of Line 13: ascending time, then increasing series index. -/
def ascNode (L : List Vertex) : List Vertex :=
  L.mergeSort (fun a b => decide (a.time < b.time) ||
    (decide (a.time = b.time) && decide (a.series ≤ b.series)))

end ITPD
