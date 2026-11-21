"""Output folder check shared by the scripts."""

from __future__ import annotations

from pathlib import Path


def check_out_dir(out_dir: Path, data_dir: Path, root: Path) -> None:
    """Refuse folders inside data/ or inside the repository (except outputs/)."""
    out, repo = out_dir.resolve(), root.resolve()
    if out.is_relative_to(data_dir.resolve()):
        raise SystemExit("Refusing to write inside the data directory (data/ is read-only).")
    if out.is_relative_to(repo) and not out.is_relative_to(repo / "outputs"):
        raise SystemExit(f"Refusing to write per-image outputs inside the repository outside outputs/: {out}")
