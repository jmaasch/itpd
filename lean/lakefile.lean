import Lake
open Lake DSL

-- Lean 4 core library only: no Mathlib and no other dependency.
package «itpd» where
  leanOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib «ITPD» where
