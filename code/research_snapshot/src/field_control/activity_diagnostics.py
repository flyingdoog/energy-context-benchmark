"""Date-balanced error allocation to recorded-zero and positive target bins."""
import numpy as np
import pandas as pd

COMPONENTS = ("mse", "mae", "bias", "zero_mse", "positive_mse",
              "zero_mae", "positive_mae", "zero_bias", "positive_bias",
              "zero_share", "positive_share")


def error_allocation(dates, target, prediction, bins):
    y = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    dates = np.asarray(dates, dtype=str)
    if y.ndim != 2 or y.shape != pred.shape or len(y) != len(dates) or len(y) == 0:
        raise ValueError("Case-aligned target, prediction and dates required")
    if not isinstance(bins, int) or bins < 1 or bins > y.shape[1]:
        raise ValueError("Invalid scored span")
    if not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError("Do not drop incomplete case targets or predictions")
    if (y < 0).any():
        raise ValueError("Negative recorded targets require a separate source audit")
    y, pred = y[:, :bins], pred[:, :bins]
    error = pred-y
    zero, positive = y == 0, y > 0
    frame = pd.DataFrame(dict(date=dates, mse=np.mean(error**2, axis=1),
        mae=np.mean(abs(error), axis=1), bias=error.mean(axis=1),
        zero_share=zero.mean(axis=1), positive_share=positive.mean(axis=1)))
    for name, mask in [("zero", zero), ("positive", positive)]:
        # Denominator is ALL case-bins, including the other target regime.
        frame[name+"_mse"] = np.mean(np.where(mask, error**2, 0), axis=1)
        frame[name+"_mae"] = np.mean(np.where(mask, abs(error), 0), axis=1)
        frame[name+"_bias"] = np.mean(np.where(mask, error, 0), axis=1)
    daily = frame.groupby("date")[list(COMPONENTS)].mean()
    daily["cases"] = frame.groupby("date").size()
    return daily.reset_index()


def summarize_allocation(daily):
    """Conditional metrics use the same date-balanced measure as total error."""
    if daily.date.duplicated().any() or len(daily) == 0:
        raise ValueError("Expected one nonempty record per date")
    result = daily[list(COMPONENTS)].mean().to_dict()
    result.update(dates=len(daily), units=int(daily.cases.sum()),
                  zero_dates=int((daily.zero_share > 0).sum()),
                  positive_dates=int((daily.positive_share > 0).sum()))
    for metric in ["mse", "mae", "bias"]:
        if not np.isclose(result["zero_"+metric]+result["positive_"+metric], result[metric], rtol=1e-9, atol=1e-9):
            raise ValueError("Components fail to recover the total error")
        for name in ["zero", "positive"]:
            share = result[name+"_share"]
            result[name+"_conditional_"+metric] = result[name+"_"+metric]/share if share else np.nan
    return result
