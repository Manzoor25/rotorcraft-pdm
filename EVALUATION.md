# Evaluation

The exam that picks the engine model is the last cycle of each FD001 test engine. That is the official C-MAPSS protocol. The test set did not get to vote during fitting. It was scored once.

## Why the NASA score is asymmetric

Remaining-life error is `predicted - true`.

- A negative error is early. The model thinks the engine will fail sooner than it will. The penalty is `exp(-error / 13) - 1`.
- A positive error is late. The model thinks the engine has more life than it does. The penalty is `exp(error / 10) - 1`.

The late branch uses 10 instead of 13, so the same-sized mistake grows faster. A late call can leave a degrading engine in service. An early call wastes a maintenance slot. Those are not equal events. RMSE treats them as equal. That is why RMSE is reported and not used to pick the winner.

A perfect call scores 0. Lower is better.

## FD001, the selection exam

| Model | RMSE | MAE | NASA score |
| --- | ---: | ---: | ---: |
| linear_regression | 38.20 | 25.87 | 43833.0 |
| random_forest (selected) | 21.33 | 15.07 | 986.4 |
| lightgbm | 23.77 | 16.33 | 1449.8 |
| xgboost | 22.91 | 15.71 | 1429.5 |
| lstm | 59.41 | 48.83 | 33094.2 |
| cnn | 39.68 | 28.01 | 115304.0 |

The random forest won. An 8-trial Optuna search moved LightGBM from 1449.8 to 1165.8. That is better, and it is still worse than the forest. The search did not get a vote to replace the selection rule after the fact. The forest remains the registered point model.

The linear model is the warning. Its RMSE is 38, which looks merely "worse." Its NASA score is 44,000, because a few late calls explode the exponential. That is the asymmetry doing its job.

## Sequence models lost

The LSTM and the 1D CNN saw the last 30 cycles. They were small, trained for 6 epochs on CPU, and scored on the same last-cycle exam. They did not beat the forest. RMSE of 59 for the LSTM means it was not just a little late. It was not ready.

I am not going to narrate that as a near miss. On this table, with this budget, the rolling-window forest is the model I would ship. A longer training run or a regime embedding might close the gap. This run did not show that. Claiming a neural net won would be false.

## Uncertainty

LightGBM quantile models at the 10th and 90th percentiles produced an interval. On the FD001 exam, 84% of true remaining-life values landed inside the interval. A perfect 10-90 interval would cover 80%. 84% is close, and slightly conservative. The maintenance policy uses the lower end, not the midpoint.

## Other subsets

FD002 and FD004 mix six operating conditions. NASA scores are larger there because there are more engines, and because a condition-blind mistake is easier. XGBoost had the best FD002 and FD004 NASA scores. LightGBM and XGBoost beat the forest on FD003. The selection rule was written as "FD001 NASA score" before those numbers were looked at. I did not switch winners after seeing FD003.

Constant sensors were dropped only where they were actually constant: six sensors on FD001, five on FD003, none on FD002 or FD004. The drop is computed, not copied from a blog post.

## Vibration anomalies

The Airbus validation file has 297 abnormal sequences out of 594. A miss is a false negative. That is the dangerous cell.

| Detector | Precision | Recall | F1 | ROC-AUC | PR-AUC | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| isolation_forest | 0.907 | 0.687 | 0.782 | 0.912 | 0.914 | 93 |
| one_class_svm | 0.855 | 0.835 | 0.845 | 0.931 | 0.947 | 49 |
| autoencoder | 0.910 | 0.677 | 0.776 | 0.913 | 0.922 | 96 |

The one-class SVM missed 49 abnormal traces. The autoencoder missed 96. The neural net did not earn the vibration job either. The SVM is the anomaly model I would show a maintainer, with the false-negative count in the same sentence as the F1 score.

The threshold for the classical detectors is the 98th percentile of healthy training scores. That choice flags about 2% of healthy flights. It is a teaching cutoff, not a certified alarm limit.

## Fleet simulation

Twelve fictional tails, `POC-01` to `POC-12`, drawn from the FD001 exam. Hidden remaining life is the test label. The fixed policy plans a pull every 80 cycles. The model policy grounds the aircraft when the cautious bound is under 15 cycles, and otherwise will not wait longer than the fixed interval.

| Policy | Unplanned failures | Availability |
| --- | ---: | ---: |
| Fixed interval | 6 of 12 | 0.50 |
| Cautious bound | 3 of 12 | 0.75 |

The model also made 8 early pulls. It bought fewer in-service failures by spending maintenance slots. That tradeoff is the result. It is not a claim about a real squadron. Twelve draws from a simulator exam are not a year of flying.

## Limitations

- C-MAPSS is a turbofan simulator. It is not a T700, and it has no gearbox. [DATA_SOURCES.md](DATA_SOURCES.md) lists the differences.
- The Airbus traces are scaled flight-test accelerometers labeled for abnormal sensor behavior, not a named crack or bearing fault.
- The sequence models were under-trained relative to a paper that spends a GPU-day. Reporting their loss is part of the evaluation, not a footnote to hide.
- The 15 and 40 cycle action cutoffs were chosen for the demonstration. A real operator would set them from the cost of a wasted pull versus the cost of an in-service failure, and then still not treat the screen as an airworthiness release.
- Interval coverage was measured on the same exam used for the point scores. A later dataset would be the honest next check.
- This system does not certify an aircraft to fly.
