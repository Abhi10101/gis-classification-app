# ============================================================
# core/ensemble.py
# ============================================================

import numpy as np


# ------------------------------------------------------------
# FUTURE (optional imports – safe if missing)
# ------------------------------------------------------------
try:
    from logger import log_event
except Exception:
    def log_event(msg):
        pass


def ensemble_probabilities(
    prob_list,
    weights=None
):
    """
    DO NOT CHANGE SIGNATURE
    """

    # --------------------------------------------------------
    # FIX: empty / invalid input protection
    # --------------------------------------------------------
    if prob_list is None or len(prob_list) == 0:
        raise ValueError("prob_list is empty")

    # --------------------------------------------------------
    # Validate probability shapes
    # --------------------------------------------------------
    ref = prob_list[0]
    if not isinstance(ref, np.ndarray) or ref.ndim != 2:
        raise ValueError("Each probability array must be 2D numpy array")

    n_samples, n_classes = ref.shape
    if n_samples <= 0 or n_classes <= 0:
        raise ValueError("Invalid probability array shape")

    for i, p in enumerate(prob_list):
        if not isinstance(p, np.ndarray):
            raise TypeError(f"Probability at index {i} is not a numpy array")
        if p.shape != ref.shape:
            raise ValueError(
                f"Probability shape mismatch at index {i}: "
                f"expected {ref.shape}, got {p.shape}"
            )
        if not np.isfinite(p).all():
            raise ValueError(
                f"Non-finite values in probability array at index {i}"
            )

    n_models = len(prob_list)

    # --------------------------------------------------------
    # FAST PATH: uniform averaging (no stacking)
    # --------------------------------------------------------
    if weights is None:
        ensemble = np.zeros_like(ref, dtype=np.float32)
        for p in prob_list:
            ensemble += p.astype(np.float32)

        ensemble /= float(n_models)

        # LOG ONLY ONCE
        if not hasattr(ensemble_probabilities, "_logged"):
            log_event("Ensemble using uniform averaging (fast path)")
            ensemble_probabilities._logged = True

        return ensemble

    # --------------------------------------------------------
    # WEIGHTED PATH (requires stacking)
    # --------------------------------------------------------
    if not isinstance(weights, (list, tuple, np.ndarray)):
        raise TypeError("weights must be list, tuple, or numpy array")

    if len(weights) != n_models:
        raise ValueError("Length of weights must match prob_list")

    weights = np.asarray(weights, dtype=np.float32)

    if not np.isfinite(weights).all():
        raise ValueError("Weights contain non-finite values")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    if weights.sum() <= 0:
        raise ValueError("Sum of weights must be > 0")

    weights = weights / weights.sum()

    # Stack once only when needed
    probs = np.stack(prob_list, axis=0).astype(np.float32)
    row_sums = probs.sum(axis=2, keepdims=True)

    if np.any(row_sums <= 0) or not np.isfinite(row_sums).all():
        raise ValueError(
            "Invalid probability distribution (sum <= 0 or non-finite)"
        )

    probs = probs / row_sums
    ensemble = np.tensordot(weights, probs, axes=(0, 0))

    # LOG ONLY ONCE
    if not hasattr(ensemble_probabilities, "_logged"):
        log_event(
            f"Ensemble using weighted averaging: {weights.tolist()}"
        )
        ensemble_probabilities._logged = True

    return ensemble.astype(np.float32)


def predict_from_ensemble(
    prob_list,
    weights=None,
    confidence_threshold=None,
    default_class=None
):
    """
    DO NOT CHANGE SIGNATURE
    """

    prob = ensemble_probabilities(prob_list, weights)

    if prob.ndim != 2:
        raise ValueError("Ensembled probability must be 2D")

    cls = prob.argmax(axis=1).astype(np.int64)
    conf = prob.max(axis=1).astype(np.float32)

    if confidence_threshold is not None:
        if default_class is None:
            raise ValueError(
                "default_class must be provided with confidence_threshold"
            )
        if not np.isfinite(confidence_threshold):
            raise ValueError("confidence_threshold must be finite")

        low = conf < confidence_threshold
        if low.any():
            cls[low] = int(default_class)

    return cls, conf
