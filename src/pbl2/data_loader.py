from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


TABLE_PREFIX = "Data Tables/"


def discover_archive(root: Path) -> Path:
    """Locate Protocol AA without modifying or extracting the source archive."""
    preferred = root / "data" / "Protocol_AA.zip"
    if preferred.exists():
        return preferred
    candidates = sorted((root / "data").glob("*.zip"))
    matches = [p for p in candidates if "protocol" in p.name.lower() or "aa" in p.name.lower()]
    if len(matches) != 1:
        raise FileNotFoundError("Expected one Protocol AA ZIP archive under data/")
    return matches[0]


def list_tables(archive: Path) -> list[str]:
    with ZipFile(archive) as zf:
        return sorted(
            Path(name).name for name in zf.namelist()
            if name.startswith(TABLE_PREFIX) and name.endswith(".txt")
        )


def read_table(archive: Path, table_name: str, **kwargs) -> pd.DataFrame:
    """Read one pipe-delimited public data table directly from the ZIP."""
    member = table_name if table_name.startswith(TABLE_PREFIX) else TABLE_PREFIX + table_name
    with ZipFile(archive) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"Table not found in archive: {member}")
        with zf.open(member) as stream:
            return pd.read_csv(stream, sep="|", low_memory=False, **kwargs)

