
# Pre-flight
python validate_sweep.py --smoke-test

# --- champS4: gorillaS4 eval after each train, split across 3 A6000s ---
python run_sweep.py --configs=configs/champS4 --gpu=0 --split=1/3 --eval --bsps=gorillaS4 --skip-existing --tier=H &
python run_sweep.py --configs=configs/champS4 --gpu=1 --split=2/3 --eval --bsps=gorillaS4 --skip-existing --tier=H &
python run_sweep.py --configs=configs/champS4 --gpu=2 --split=3/3 --eval --bsps=gorillaS4 --skip-existing --tier=H
wait

# --- champTa: gorillaTa eval ---
python run_sweep.py --configs=configs/champTa --gpu=0 --split=1/3 --eval --bsps=gorillaTa --skip-existing --tier=H &
python run_sweep.py --configs=configs/champTa --gpu=1 --split=2/3 --eval --bsps=gorillaTa --skip-existing --tier=H &
python run_sweep.py --configs=configs/champTa --gpu=2 --split=3/3 --eval --bsps=gorillaTa --skip-existing --tier=H
wait

# --- Second eval pass: hawk{X} (training skipped via --skip-existing, eval runs) ---
python run_sweep.py --configs=configs/champS4 --gpu=0 --eval --bsps=hawkS4 --skip-existing --tier=H
python run_sweep.py --configs=configs/champTa --gpu=0 --eval --bsps=hawkTa --skip-existing --tier=H

# --- Read-out ---
python scripts/registry_query.py top --bsps=hawkTa  --limit=10
python scripts/registry_query.py top --bsps=hawkS4  --limit=10
python scripts/registry_query.py top --bsps=gorillaTa --limit=10
python scripts/registry_query.py top --bsps=gorillaS4 --limit=10
