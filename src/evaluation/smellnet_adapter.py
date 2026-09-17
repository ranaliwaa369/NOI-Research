"""Auditable SmellNet-Base ingestion for the NOI v0.4 research track.

The adapter deliberately stops at validated physical-sensor recordings.  It
does not train a model, select an open-set threshold, or claim real-world
performance.  Ingredient and family labels are retained as scoring metadata
and are never appended to the six sensor channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import math
from pathlib import Path
from typing import Iterable, TypeAlias

import pandas as pd


SensorRow: TypeAlias = tuple[float, ...]
SensorMatrix: TypeAlias = tuple[SensorRow, ...]

SMELLNET_SENSOR_COLUMNS = (
    "NO2",
    "C2H5OH",
    "VOC",
    "CO",
    "Alcohol",
    "LPG",
)

SMELLNET_INGREDIENT_TO_FAMILY = {
    "peanuts": "nuts",
    "cashew": "nuts",
    "chestnuts": "nuts",
    "pistachios": "nuts",
    "almond": "nuts",
    "hazelnut": "nuts",
    "walnuts": "nuts",
    "pecans": "nuts",
    "brazil_nut": "nuts",
    "pili_nut": "nuts",
    "cumin": "spices",
    "star_anise": "spices",
    "nutmeg": "spices",
    "cloves": "spices",
    "ginger": "spices",
    "allspice": "spices",
    "chervil": "spices",
    "mustard": "spices",
    "cinnamon": "spices",
    "saffron": "spices",
    "angelica": "herbs",
    "garlic": "herbs",
    "chives": "herbs",
    "turnip": "herbs",
    "dill": "herbs",
    "mugwort": "herbs",
    "chamomile": "herbs",
    "coriander": "herbs",
    "oregano": "herbs",
    "mint": "herbs",
    "kiwi": "fruits",
    "pineapple": "fruits",
    "banana": "fruits",
    "lemon": "fruits",
    "mandarin_orange": "fruits",
    "strawberry": "fruits",
    "apple": "fruits",
    "mango": "fruits",
    "peach": "fruits",
    "pear": "fruits",
    "cauliflower": "vegetables",
    "brussel_sprouts": "vegetables",
    "broccoli": "vegetables",
    "sweet_potato": "vegetables",
    "asparagus": "vegetables",
    "avocado": "vegetables",
    "radish": "vegetables",
    "tomato": "vegetables",
    "potato": "vegetables",
    "cabbage": "vegetables",
}


class SmellNetAdapterError(ValueError):
    """Raised when a SmellNet recording or directory is not auditable."""


class SmellNetSplit(str, Enum):
    """Official SmellNet-Base directory partitions."""

    TRAINING = "training"
    TESTING = "testing"


@dataclass(frozen=True, slots=True)
class SmellNetRecording:
    """One validated physical-sensor time series and its scoring metadata."""

    recording_id: str
    split: SmellNetSplit
    ingredient: str
    family: str
    source_path: str
    sensor_columns: tuple[str, ...]
    sensor_rows: SensorMatrix
    source_sha256: str

    def __post_init__(self) -> None:
        if not self.recording_id.strip():
            raise SmellNetAdapterError("recording_id must be nonempty.")
        if not isinstance(self.split, SmellNetSplit):
            raise SmellNetAdapterError("split must be a SmellNetSplit value.")
        if not self.ingredient.strip() or not self.family.strip():
            raise SmellNetAdapterError(
                "ingredient and family must be nonempty."
            )
        if self.sensor_columns != SMELLNET_SENSOR_COLUMNS:
            raise SmellNetAdapterError(
                "sensor_columns must match the six locked SmellNet channels."
            )
        if not self.sensor_rows:
            raise SmellNetAdapterError(
                "A sensor recording must contain at least one row."
            )
        for row in self.sensor_rows:
            if len(row) != len(SMELLNET_SENSOR_COLUMNS):
                raise SmellNetAdapterError(
                    "Every sensor row must contain exactly six values."
                )
            if any(not math.isfinite(value) for value in row):
                raise SmellNetAdapterError(
                    "Sensor rows must contain only finite values."
                )
        if len(self.source_sha256) != 64:
            raise SmellNetAdapterError(
                "source_sha256 must be a hexadecimal SHA-256 digest."
            )
        try:
            int(self.source_sha256, 16)
        except ValueError as error:
            raise SmellNetAdapterError(
                "source_sha256 must be a hexadecimal SHA-256 digest."
            ) from error


@dataclass(frozen=True, slots=True)
class SmellNetAuditReport:
    """Read-only integrity summary for one local SmellNet-Base copy."""

    passed: bool
    recording_count: int
    training_count: int
    testing_count: int
    ingredient_count: int
    family_count: int
    duplicate_recording_ids: tuple[str, ...]
    cross_split_content_duplicates: tuple[str, ...]


def load_smellnet_recording(
    csv_path: str | Path,
    *,
    split: SmellNetSplit,
    ingredient: str | None = None,
) -> SmellNetRecording:
    """Load one CSV while keeping labels outside the sensor feature matrix."""

    path = Path(csv_path)
    if not path.is_file():
        raise SmellNetAdapterError(
            f"SmellNet recording does not exist: {path}"
        )
    if path.suffix.lower() != ".csv":
        raise SmellNetAdapterError("SmellNet recordings must be CSV files.")
    if not isinstance(split, SmellNetSplit):
        raise SmellNetAdapterError("split must be a SmellNetSplit value.")

    resolved_ingredient = ingredient or path.parent.name
    if resolved_ingredient not in SMELLNET_INGREDIENT_TO_FAMILY:
        raise SmellNetAdapterError(
            f"Unknown SmellNet ingredient: {resolved_ingredient}"
        )

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise SmellNetAdapterError(
            f"Could not read SmellNet recording: {path}"
        ) from error

    missing_columns = tuple(
        column
        for column in SMELLNET_SENSOR_COLUMNS
        if column not in frame.columns
    )
    if missing_columns:
        raise SmellNetAdapterError(
            "Missing locked SmellNet sensor columns: "
            + ", ".join(missing_columns)
        )
    if frame.empty:
        raise SmellNetAdapterError("SmellNet recording is empty.")

    sensor_frame = frame.loc[:, SMELLNET_SENSOR_COLUMNS]
    try:
        numeric_frame = sensor_frame.apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise SmellNetAdapterError(
            "Sensor channels must contain only numeric values."
        ) from error

    sensor_rows = tuple(
        tuple(float(value) for value in row)
        for row in numeric_frame.itertuples(index=False, name=None)
    )
    if any(not math.isfinite(value) for row in sensor_rows for value in row):
        raise SmellNetAdapterError(
            "Sensor channels must contain only finite values."
        )

    source_digest = sha256(path.read_bytes()).hexdigest()
    recording_id = (
        f"{split.value}/{resolved_ingredient}/{path.name}"
    )

    return SmellNetRecording(
        recording_id=recording_id,
        split=split,
        ingredient=resolved_ingredient,
        family=SMELLNET_INGREDIENT_TO_FAMILY[resolved_ingredient],
        source_path=str(path),
        sensor_columns=SMELLNET_SENSOR_COLUMNS,
        sensor_rows=sensor_rows,
        source_sha256=source_digest,
    )


def discover_smellnet_base(
    dataset_root: str | Path,
) -> tuple[SmellNetRecording, ...]:
    """Discover official training/testing CSVs in deterministic order."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise SmellNetAdapterError(
            f"SmellNet dataset root does not exist: {root}"
        )

    recordings: list[SmellNetRecording] = []
    for split in (SmellNetSplit.TRAINING, SmellNetSplit.TESTING):
        split_path = root / split.value
        if not split_path.is_dir():
            raise SmellNetAdapterError(
                f"Missing official SmellNet split directory: {split_path}"
            )

        for ingredient_path in sorted(split_path.iterdir()):
            if not ingredient_path.is_dir():
                continue
            ingredient = ingredient_path.name
            if ingredient not in SMELLNET_INGREDIENT_TO_FAMILY:
                raise SmellNetAdapterError(
                    f"Unknown SmellNet ingredient directory: {ingredient}"
                )
            for csv_path in sorted(ingredient_path.glob("*.csv")):
                recordings.append(
                    load_smellnet_recording(
                        csv_path,
                        split=split,
                        ingredient=ingredient,
                    )
                )

    if not recordings:
        raise SmellNetAdapterError(
            "No SmellNet CSV recordings were discovered."
        )
    return tuple(recordings)


def audit_smellnet_recordings(
    recordings: Iterable[SmellNetRecording],
    *,
    raise_on_failure: bool = False,
) -> SmellNetAuditReport:
    """Detect duplicate identities and byte-identical cross-split files."""

    values = tuple(recordings)
    if not values:
        raise SmellNetAdapterError("At least one recording is required.")
    if any(not isinstance(value, SmellNetRecording) for value in values):
        raise SmellNetAdapterError(
            "recordings must contain only SmellNetRecording values."
        )

    id_counts: dict[str, int] = {}
    digest_splits: dict[str, set[SmellNetSplit]] = {}
    for recording in values:
        id_counts[recording.recording_id] = (
            id_counts.get(recording.recording_id, 0) + 1
        )
        digest_splits.setdefault(recording.source_sha256, set()).add(
            recording.split
        )

    duplicate_ids = tuple(
        sorted(key for key, count in id_counts.items() if count > 1)
    )
    cross_split_duplicates = tuple(
        sorted(
            digest
            for digest, splits in digest_splits.items()
            if len(splits) > 1
        )
    )
    passed = not duplicate_ids and not cross_split_duplicates

    report = SmellNetAuditReport(
        passed=passed,
        recording_count=len(values),
        training_count=sum(
            value.split is SmellNetSplit.TRAINING for value in values
        ),
        testing_count=sum(
            value.split is SmellNetSplit.TESTING for value in values
        ),
        ingredient_count=len({value.ingredient for value in values}),
        family_count=len({value.family for value in values}),
        duplicate_recording_ids=duplicate_ids,
        cross_split_content_duplicates=cross_split_duplicates,
    )

    if raise_on_failure and not report.passed:
        raise SmellNetAdapterError(
            "SmellNet integrity audit failed: duplicate IDs or "
            "byte-identical files cross the official split boundary."
        )
    return report
