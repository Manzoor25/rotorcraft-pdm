# Results

Written by scripts/write_results.py from reports/metrics.json.
The FD001 winner is the lowest NASA score. Lower is better.

## FD001 classical models

| Model | RMSE | MAE | NASA score |
| --- | ---: | ---: | ---: |
| linear_regression | 38.20 | 25.87 | 43833.0 |
| random_forest (selected) | 21.33 | 15.07 | 986.4 |
| lightgbm | 23.77 | 16.33 | 1449.8 |
| xgboost | 22.91 | 15.71 | 1429.5 |

## Sequence models on FD001

| Model | RMSE | MAE | NASA score |
| --- | ---: | ---: | ---: |
| lstm | 59.41 | 48.83 | 33094.2 |
| cnn | 39.68 | 28.01 | 115304.0 |

10-90 interval coverage on the FD001 exam: 0.840

## Other subsets

| Subset | Model | RMSE | NASA score |
| --- | --- | ---: | ---: |
| FD002 | linear_regression | 38.40 | 779032.7 |
| FD002 | random_forest | 19.99 | 9582.8 |
| FD002 | lightgbm | 19.05 | 10115.4 |
| FD002 | xgboost | 19.94 | 7674.7 |
| FD003 | linear_regression | 31.06 | 63074.0 |
| FD003 | random_forest | 22.92 | 2315.0 |
| FD003 | lightgbm | 22.25 | 1386.5 |
| FD003 | xgboost | 21.85 | 1342.8 |
| FD004 | linear_regression | 28.12 | 55376.8 |
| FD004 | random_forest | 17.98 | 1792.5 |
| FD004 | lightgbm | 17.92 | 1937.2 |
| FD004 | xgboost | 17.50 | 1715.1 |
