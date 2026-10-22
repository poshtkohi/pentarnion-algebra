"""Read a reference CSV directly, or its single-member ZIP, without extraction.

The distributed data are immutable inputs. This helper never writes into data/.
"""
from __future__ import annotations
from contextlib import contextmanager
import io
from pathlib import Path
from typing import Iterator, TextIO
import zipfile


def _archive_member(archive: zipfile.ZipFile, expected: str) -> str:
    members = [i for i in archive.infolist() if not i.is_dir()]
    if len(members) != 1 or Path(members[0].filename).name != expected:
        raise ValueError(f"Expected one CSV named {expected!r} in {archive.filename}")
    return members[0].filename


@contextmanager
def open_reference_text(path: str | Path) -> Iterator[TextIO]:
    """Open UTF-8 text, falling back from file.csv to file.zip."""
    requested = Path(path)
    if requested.is_file():
        with requested.open('r', encoding='utf-8-sig', newline='') as handle:
            yield handle
        return
    archive_path = requested.with_suffix('.zip')
    if requested.suffix != '.csv' or not archive_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {requested} (nor {archive_path})")
    with zipfile.ZipFile(archive_path) as archive:
        member = _archive_member(archive, requested.name)
        with archive.open(member) as binary:
            with io.TextIOWrapper(binary, encoding='utf-8-sig', newline='') as handle:
                yield handle


def read_reference_csv(path: str | Path, *args, **kwargs):
    """pandas.read_csv wrapper; lazy readers are deliberately unsupported."""
    import pandas as pd
    if kwargs.get('chunksize') is not None or kwargs.get('iterator'):
        raise ValueError('Use open_reference_text for streamed or chunked reading')
    with open_reference_text(path) as handle:
        return pd.read_csv(handle, *args, **kwargs)
