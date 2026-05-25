
# =============================================================================
# Anchored SAE sweep on champTa / fc1 / tigerTa  (2026-05-22)
# =============================================================================
#   I-series: anchored-jumprelu  (t=64, exp=8) × λ_high ∈ {0.03, 0.10, 0.30, 1.0} × 3 seeds
#   J-series: anchored-batchtopk (k=16,  exp=8) × λ_high ∈ {0.03, 0.10, 0.30, 1.0} × 3 seeds
# Total: 24 runs.  Hardware: 3× A6000.
# Anchor: first 36 dictionary slots → tigerTa BSPs (prefix mapping), BCE on
# pre-activation z. High-tier categories (λ_high): tiger_square_winnable,
# tiger_offered_completing_attr, tiger_line_winnable. Other 13 BSPs get
# λ_medium = λ_high / 3.
# =============================================================================

# Pre-flight
# python validate_sweep.py --smoke-test

# --- Train: I-tier + J-tier split across 3 A6000s ---
python run_sweep.py --configs=configs/champTa --gpu=0 --split=1/3 --eval --bsps=tigerTa --skip-existing --tier=I,J &
python run_sweep.py --configs=configs/champTa --gpu=1 --split=2/3 --eval --bsps=tigerTa --skip-existing --tier=I,J &
python run_sweep.py --configs=configs/champTa --gpu=2 --split=3/3 --eval --bsps=tigerTa --skip-existing --tier=I,J
wait

# --- Second eval pass: gorillaTa + hawkTa (training already done, --skip-existing) ---
# Confirms anchoring did not regress general capacity on the standard bases.
python run_sweep.py --configs=configs/champTa --gpu=0 --eval --bsps=gorillaTa --skip-existing --tier=I,J &
python run_sweep.py --configs=configs/champTa --gpu=1 --eval --bsps=hawkTa    --skip-existing --tier=I,J
wait

# --- Read-out: anchored diagonal lives inside the registry entry's metrics ---
python scripts/registry_query.py top --bsps=tigerTa  --limit=20
python scripts/registry_query.py top --bsps=gorillaTa --limit=20
python scripts/registry_query.py top --bsps=hawkTa    --limit=20

# Sanity: compare a low-λ vs high-λ run side by side
python scripts/registry_query.py compare \
    I01-champTa-lh003-s42-anchored-jumprelu-t64-exp8-s4.fc1 \
    I04-champTa-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1 \
    --bsps=tigerTa
