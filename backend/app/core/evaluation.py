"""Splitting and interval helpers shared by every evaluation script.

These live in one place on purpose. If ``train_model.py`` and
``evaluate_engines.py`` each defined their own split, the numbers they publish
would be measured on different data while looking directly comparable, which is
precisely the kind of quiet mismatch this rework exists to remove. Every script
that reports a number calls :func:`three_way_group_split` and gets the same
rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from app.core.paths import DATASET_PATH
from app.utils.text_cleaning import clean_text

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_FRACTION_FOLDS = 5  # one fifth of the groups become the test set
VALIDATION_FRACTION_FOLDS = 4  # then one quarter of the rest becomes validation


@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def development(self) -> pd.DataFrame:
        """train + validation: everything the shipped model may learn from."""
        return pd.concat([self.train, self.validation])


def load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATASET_PATH)
    dataset["cleaned_text"] = dataset["text"].apply(clean_text)
    dataset["source_section_key"] = dataset["source_name"] + " :: " + dataset["source_section"]
    return dataset


def group_holdout(frame: pd.DataFrame, n_folds: int, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve 1/n_folds of the *groups* out of ``frame``, stratified by label."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    remainder_index, holdout_index = next(
        splitter.split(frame["cleaned_text"], frame["disease"], groups=frame["group_id"])
    )
    return frame.iloc[remainder_index].copy(), frame.iloc[holdout_index].copy()


def three_way_group_split(dataset: pd.DataFrame) -> DataSplit:
    """The one split every script in this project uses.

    Groups, not rows. A paraphrase and the row it was derived from are
    near-duplicates; a row-wise split puts one in train and the other in test
    and then congratulates the model for recognising it.
    """
    development, test = group_holdout(dataset, TEST_FRACTION_FOLDS)
    train, validation = group_holdout(development, VALIDATION_FRACTION_FOLDS)
    assert_no_group_overlap(train, validation, test)
    return DataSplit(train=train, validation=validation, test=test)


def assert_no_group_overlap(*frames: pd.DataFrame) -> None:
    """A split that leaks a group is worse than no split, so fail loudly."""
    seen: set[str] = set()
    for frame in frames:
        groups = set(frame["group_id"])
        overlap = seen & groups
        if overlap:
            raise RuntimeError(f"Group leak across splits: {sorted(overlap)[:5]}")
        seen |= groups


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation because per-class supports here
    are in the teens, where the normal interval misbehaves and can run outside
    [0, 1] at proportions near 0 or 1.
    """
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    return (max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator))


def t_interval(values: list[float], z: float = 2.776) -> tuple[float, float]:
    """95% interval across CV folds. Default z is t(0.975, df=4), i.e. 5 folds."""
    if not values:
        return (0.0, 0.0)
    if len(values) < 2:
        return (float(values[0]), float(values[0]))
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return (max(0.0, mean - z * standard_error), min(1.0, mean + z * standard_error))
