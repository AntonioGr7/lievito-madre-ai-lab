from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from datasets import DatasetDict, load_dataset, load_from_disk


@runtime_checkable
class DataSource(Protocol):
    def load(self) -> DatasetDict: ...


@dataclass
class HFSource:
    """Load a dataset directly from the Hugging Face Hub."""

    dataset_id: str
    config_name: str | None = None
    kwargs: dict = field(default_factory=dict)

    def load(self) -> DatasetDict:
        return load_dataset(self.dataset_id, self.config_name, **self.kwargs)


@dataclass
class LocalSource:
    """Load a dataset from disk.

    format='arrow'   → expects a folder saved with dataset.save_to_disk()
    format='csv'     → expects {'train': 'path/train.csv', 'test': 'path/test.csv', ...}
    format='json'    → same as csv but .jsonl files
    format='parquet' → same as csv but .parquet files
    """

    path: str | Path
    format: str = "arrow"
    split_files: dict[str, str] | None = None  # required for csv / json / parquet

    def load(self) -> DatasetDict:
        path = Path(self.path)
        if self.format == "arrow":
            return load_from_disk(str(path))
        if self.split_files is None:
            raise ValueError(
                f"Provide split_files={{split: filepath}} when format='{self.format}'"
            )
        return load_dataset(self.format, data_files=self.split_files)


@dataclass
class DriveSource:
    """Download a file or folder from Google Drive via gdown, then load as LocalSource.

    Requires: pip install gdown

    file_id  : the ID from the shareable link  (…/d/<file_id>/view)
    dest     : local path where the file will be saved / extracted
    format   : how to open the result once downloaded (passed to LocalSource)
    is_folder: set True to download an entire Drive folder with gdown
    """

    file_id: str
    dest: str | Path
    format: str = "arrow"
    is_folder: bool = False

    def load(self) -> DatasetDict:
        try:
            import gdown
        except ImportError as e:
            raise ImportError("Install gdown first:  pip install gdown") from e

        dest = Path(self.dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if self.is_folder:
            url = f"https://drive.google.com/drive/folders/{self.file_id}"
            gdown.download_folder(url, output=str(dest), quiet=False)
        else:
            url = f"https://drive.google.com/uc?id={self.file_id}"
            gdown.download(url, str(dest), quiet=False)

        return LocalSource(dest, format=self.format).load()
