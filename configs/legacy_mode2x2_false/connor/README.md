# Connor Experiments

**Goal:** Push past the ~0.21 coverage ceiling seen in Arnold/Arnold Beta.

**Key findings from Arnold/Arnold Beta:**
- JumpReLU wins: t32 → best coverage (0.2137), t64 → best board recon (0.5183 frac reconstructable)
- ~85-90% dead features is the main bottleneck for all sparse archs
- BatchTopK k32 competitive with variable L0
- Vanilla has good coverage but poor board reconstruction — not useful

**Connor strategy:**
1. JumpReLU t16 — push sparser, crisper per-BSP features
2. JumpReLU t48 — fill the t32→t64 gap
3. JumpReLU t32 + exp16 — 2× features to kill dead feature problem
4. JumpReLU t64 + exp16 — best board-recon target + 2× expansion
5. BatchTopK k16 — variable-L0 sparser regime
6. BatchTopK k64 — variable-L0 scaled up from winning k32

| Run ID | Arch | Key Param | Rationale |
|--------|------|-----------|-----------|
| connor-jumprelu-t16-exp8-fc1 | jumprelu | t=16 | Sparser than best t32 |
| connor-jumprelu-t48-exp8-fc1 | jumprelu | t=48 | Gap between t32 and t64 |
| connor-jumprelu-t32-exp16-fc1 | jumprelu | t=32, exp=16 | Best coverage target + 2× expansion |
| connor-jumprelu-t64-exp16-fc1 | jumprelu | t=64, exp=16 | Best board-recon target + 2× expansion |
| connor-batchtopk-k16-exp8-fc1 | batchtopk | k=16 | Variable-L0 sparser |
| connor-batchtopk-k64-exp8-fc1 | batchtopk | k=64 | Variable-L0 scaled up |
