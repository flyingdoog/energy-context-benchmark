# Fixed-model versus retrained reassignment

MSE / MAE, original group-total units, averaged over all native spans and dates.
Both permuted arms use the same three cached mappings. Fixed permutation keeps the
original Tree; retrained reassignment uses the existing fixed-recipe retrained models.
The mapping moves auxiliary values and masks together within the original query population.

| Source / site | Period | Power only | Paired | Fixed-model reassignment | Retrained reassignment |
|---|---|---:|---:|---:|---:|
| ACN / Caltech | autumn | 8.529 / 1.725 | 8.313 / 1.687 | 9.663 / 1.948 | 8.408 / 1.722 |
| ACN / Caltech | validation | 6.823 / 1.603 | 6.148 / 1.510 | 7.376 / 1.762 | 6.574 / 1.567 |
| ACN / Caltech | winter | 7.772 / 1.679 | 6.827 / 1.610 | 9.376 / 1.969 | 8.040 / 1.717 |
| ACN / Office | autumn | 13.029 / 2.416 | 11.349 / 2.292 | 12.343 / 2.552 | 12.341 / 2.374 |
| ACN / Office | validation | 6.711 / 1.735 | 5.748 / 1.631 | 6.460 / 1.886 | 6.494 / 1.732 |
| ACN / Office | winter | 19.064 / 2.909 | 15.676 / 2.697 | 16.044 / 2.837 | 17.738 / 2.856 |
| BPA / Field study | spring | 42.583 / 4.238 | 38.327 / 3.904 | 60.116 / 5.173 | 44.105 / 4.302 |
| BPA / Field study | summer | 26.850 / 3.625 | 26.277 / 3.400 | 45.256 / 4.529 | 33.780 / 3.819 |
| BPA / Field study | validation | 33.202 / 3.595 | 23.816 / 3.040 | 35.384 / 4.078 | 23.771 / 3.137 |

## Group-MSE penalty relative to correctly paired Tree

All percentages share the paired-error denominator. Relearning recovery = fixed penalty - retrained penalty.
Intervals: 1,500 paired calendar-week draws, conditional on the fitted models.

| Site | Period | Fixed-model penalty (%) [95% CI] | Retrained penalty (%) [95% CI] | Relearning recovery (percentage points) |
|---|---|---:|---:|---:|
| Caltech | autumn | +16.24 [+1.34, +32.26] | +1.15 [-5.86, +9.69] | +15.09 |
| Caltech | validation | +19.96 [+16.14, +23.98] | +6.92 [+1.62, +15.46] | +13.04 |
| Caltech | winter | +37.33 [+29.73, +43.12] | +17.76 [+6.80, +27.94] | +19.57 |
| Office | autumn | +8.76 [+2.95, +15.31] | +8.75 [+4.64, +15.01] | +0.01 |
| Office | validation | +12.39 [+6.83, +19.17] | +12.98 [+5.39, +20.86] | -0.59 |
| Office | winter | +2.35 [-0.89, +5.29] | +13.16 [+5.60, +19.92] | -10.81 |
| Field study | spring | +56.85 [+39.75, +73.11] | +15.08 [+9.30, +21.70] | +41.77 |
| Field study | summer | +72.23 [+53.74, +94.08] | +28.55 [+23.86, +35.08] | +43.67 |
| Field study | validation | +48.57 [+38.98, +62.31] | -0.19 [-6.22, +4.03] | +48.76 |

Fixed-model error sensitivity and recoverability after retraining are different quantities.
This is a model-conditional benchmark control, not a new permutation-importance estimator,
not conditional sampling given every protected variable, and not an identified causal effect.
Validation and previously inspected later periods are retained. CU and EPFL are not newly
evaluated here. Stronger ordinary validation references remain in the diagnostic-value readout.
