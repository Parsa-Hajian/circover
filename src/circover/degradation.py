"""
Controlled degradation-and-recovery benchmark (Chapter 14).

Protocol:
  1. Remove minority-class samples in `steps` equal increments (0% → 100%).
  2. At each level, apply an oversampler (or any sklearn-compatible estimator)
     and measure classification metrics via cross-validation.
  3. Return a DataFrame of (degradation_level, metric_value) for plotting.

Compatible with any imblearn/sklearn pipeline or estimator.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import check_random_state

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class DegradationBench:
    """
    Controlled degradation-and-recovery benchmark.

    Parameters
    ----------
    steps : int
        Number of degradation steps (including 0 = no degradation).
    metric : str
        Scoring metric passed to ``cross_val_score`` (e.g. ``"f1"``,
        ``"roc_auc"``, ``"f1_macro"``).
    cv : int
        Number of cross-validation folds.
    random_state : int or None
        Random seed for reproducibility.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from imblearn.pipeline import Pipeline
    >>> import circover as cc
    >>> pipe = Pipeline([("over", cc.GVMCO()), ("clf", RandomForestClassifier())])
    >>> bench = cc.DegradationBench(steps=10, metric="f1")
    >>> results = bench.run(pipe, X, y)
    >>> bench.plot(results)
    """

    def __init__(
        self,
        steps: int = 10,
        metric: str = "f1",
        cv: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.steps = steps
        self.metric = metric
        self.cv = cv
        self.random_state = random_state

    # ------------------------------------------------------------------
    def run(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "pd.DataFrame | dict":
        """
        Run the degradation benchmark.

        Parameters
        ----------
        estimator : sklearn-compatible estimator or pipeline
        X : (n, d) feature array
        y : (n,) label array (binary: minority class = 1)

        Returns
        -------
        DataFrame (or dict if pandas is unavailable) with columns:
            ``degradation``, ``score``, ``n_minority``
        """
        rng = check_random_state(self.random_state)
        X, y = np.asarray(X), np.asarray(y)

        minority_mask = y == 1
        minority_idx = np.where(minority_mask)[0]
        n_minority = len(minority_idx)

        degradation_levels = np.linspace(0.0, 1.0, self.steps + 1)
        records = []

        for delta in degradation_levels:
            n_remove = int(round(delta * n_minority))
            if n_remove >= n_minority:
                # Fully degraded — no minority samples remain; score = 0
                records.append({
                    "degradation": float(delta),
                    "score": 0.0,
                    "n_minority": 0,
                })
                continue

            # Remove n_remove randomly selected minority points
            remove_idx = rng.choice(minority_idx, size=n_remove, replace=False)
            keep_mask = np.ones(len(y), dtype=bool)
            keep_mask[remove_idx] = False
            X_deg, y_deg = X[keep_mask], y[keep_mask]

            n_remaining = int(minority_mask.sum()) - n_remove

            try:
                est = clone(estimator)
                scores = cross_val_score(
                    est, X_deg, y_deg,
                    scoring=self.metric,
                    cv=StratifiedKFold(
                        n_splits=min(self.cv, n_remaining),
                        shuffle=True,
                        random_state=rng.randint(0, 2**31),
                    ),
                    error_score=0.0,
                )
                mean_score = float(np.mean(scores))
            except Exception:
                mean_score = 0.0

            records.append({
                "degradation": float(delta),
                "score": mean_score,
                "n_minority": n_remaining,
            })

        if _HAS_PANDAS:
            import pandas as pd
            return pd.DataFrame(records)
        return records

    # ------------------------------------------------------------------
    def plot(
        self,
        results,
        baseline_results=None,
        title: str = "Degradation–Recovery Curve",
        ax=None,
    ):
        """
        Plot recovery curves.

        Parameters
        ----------
        results : DataFrame or list of dicts returned by ``run()``
        baseline_results : optional DataFrame/list for a no-oversampling baseline
        title : str
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for DegradationBench.plot(). "
                "Install it with: pip install matplotlib"
            ) from exc

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.get_figure()

        def _extract(r):
            if _HAS_PANDAS:
                import pandas as pd
                if isinstance(r, pd.DataFrame):
                    return r["degradation"].values, r["score"].values
            # list of dicts
            deltas = [d["degradation"] for d in r]
            scores = [d["score"] for d in r]
            return np.array(deltas), np.array(scores)

        deltas, scores = _extract(results)
        ax.plot(deltas, scores, "o-", color="#003282", linewidth=2, label="method")
        ax.fill_between(deltas, scores, alpha=0.12, color="#003282")

        if baseline_results is not None:
            bd, bs = _extract(baseline_results)
            ax.plot(bd, bs, "--", color="#888888", linewidth=1.5, label="baseline")

        # Area under recovery curve (ARI)
        ari = float(np.trapz(scores, deltas))
        ax.set_title(f"{title}  (ARI = {ari:.3f})", fontsize=11)
        ax.set_xlabel("Degradation level δ")
        ax.set_ylabel(f"CV {self.metric}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @staticmethod
    def area_recovery_index(results) -> float:
        """
        Compute the Area Recovery Index (ARI) = ∫ score(δ) dδ.

        Parameters
        ----------
        results : DataFrame or list of dicts returned by ``run()``

        Returns
        -------
        float
        """
        if _HAS_PANDAS:
            import pandas as pd
            if isinstance(results, pd.DataFrame):
                return float(np.trapz(
                    results["score"].values,
                    results["degradation"].values,
                ))
        scores = [d["score"] for d in results]
        deltas = [d["degradation"] for d in results]
        return float(np.trapz(scores, deltas))
