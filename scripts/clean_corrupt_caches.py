"""Delete corrupt torch .pt files (e.g. eval _h caches truncated by a crash or
concurrent write). A torch checkpoint is a zip archive; if its central
directory is unreadable the file is corrupt ("PytorchStreamReader ... failed
finding central directory"). This checks that cheaply (no tensor load) and
deletes the bad ones so the next run regenerates them.

Usage:
    clean_corrupt_caches.py <path>... [--dry-run] [--quiet]

Arguments:
    <path>        Files, directories, or globs to scan. Directories are scanned
                  for *.pt recursively.

Options:
    --dry-run     Report corrupt files but do not delete.
    --quiet       Only print the summary line.
    -h --help     Show this help.
"""

import glob
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docopt import docopt


def is_corrupt(path: Path) -> bool:
    """True if the .pt is not a readable zip (central directory unreadable).

    Fast: opening a ZipFile reads only the end-of-file central directory; we do
    NOT call testzip() (which would decompress GB-scale caches)."""
    try:
        with zipfile.ZipFile(path) as z:
            z.namelist()
        return False
    except Exception:
        return True


def collect(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pt")))
        elif any(ch in raw for ch in "*?["):
            out.extend(sorted(Path(m) for m in glob.glob(raw)))
        elif p.exists():
            out.append(p)
    # de-dup, keep order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main():
    args = docopt(__doc__)
    quiet, dry = args["--quiet"], args["--dry-run"]
    files = collect(args["<path>"])
    corrupt = [p for p in files if is_corrupt(p)]

    for p in corrupt:
        if dry:
            if not quiet:
                print(f"[corrupt] {p.as_posix()}  (would delete)")
        else:
            try:
                p.unlink()
                if not quiet:
                    print(f"[deleted] {p.as_posix()}")
            except OSError as e:
                print(f"[ERROR ] could not delete {p.as_posix()}: {e}", file=sys.stderr)

    action = "would delete" if dry else "deleted"
    print(f"[clean_corrupt_caches] scanned {len(files)}, {action} {len(corrupt)} corrupt")


if __name__ == "__main__":
    main()
