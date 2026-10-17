I built a predictive-maintenance proof of concept for rotorcraft using only public data. It is not an Apache, and it is not affiliated with the U.S. Army or Boeing.

The engine side uses NASA C-MAPSS, a turbofan run-to-failure simulator, as a stand-in for turboshaft wear. The vibration side uses Airbus flight-test accelerometer traces. I refused to invent an operational dataset.

Four classical models and two small sequence nets predicted remaining life. The selection metric was the NASA score, which penalizes a late "you are fine" call more than an early pull. A random forest won on the FD001 exam (NASA score 986, RMSE 21). The LSTM and the 1D CNN did not. An 8-trial search improved LightGBM and still did not beat the forest.

On vibration, a one-class SVM missed 49 of 297 abnormal traces. The autoencoder missed 96. I shipped the model that missed fewer, and I say the miss count in the same sentence as the F1 score.

A 12-aircraft simulation cut unplanned failures from 6 to 3 versus a fixed 80-cycle pull, by spending more early maintenance. That is decision support, not an airworthiness release.

Code, decisions, and the evaluation write-up: https://github.com/Manzoor25/rotorcraft-pdm
