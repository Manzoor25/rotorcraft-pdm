# Data sources and licenses

Nothing in this project is military, controlled, or classified. If a file required a request, a login, or a click-through that blocked a scripted download, it was not used, and the substitution is written here.

## NASA C-MAPSS (used)

- What it is: simulated turbofan engines run until failure. Four subsets, FD001 to FD004.
- Official file: `CMAPSSData.zip` from the NASA Open Data legacy path `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`.
- Catalog page: https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data
- Also described by the NASA Prognostics Center of Excellence data repository.
- License: U.S. Government work. The catalog marks NASA data as public. Redistribution of derived metrics is permitted. This repository does not commit the zip. `scripts/download_data.py` fetches it.
- Citation, required by the file's own readme: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," PHM08, Denver, 2008.

### Why it is a reasonable proxy for a T700-class turboshaft

A turboshaft and a turbofan are relatives. Both have a compressor, a combustor, and a turbine. Both degrade by the same physical habits: fouling, erosion, and hot-section wear that show up as temperature and speed drifting away from a healthy baseline. C-MAPSS gives run-to-failure multivariate sensor traces with a known end of life. That is the shape a remaining-life model needs, and no public turboshaft file has it.

### Where the proxy is wrong

- A T700 drives a rotor through a gearbox. C-MAPSS models a turbofan making thrust. There is no rotor, no mast, no tail-rotor drive.
- C-MAPSS "cycles" are flight cycles in a simulator, not helicopter flight hours.
- The fault modes are fan and HPC degradation in a turbofan map, not the T700's particle-separator or gearbox-driven load swings.
- Operational settings in FD002 and FD004 are altitude and throttle, not hover versus forward flight.
- Using this file to claim an AH-64 engine prognosis would be false. This project does not claim that.

## N-CMAPSS (not used)

The newer N-CMAPSS set is larger and, as of this writing, is not available as a stable anonymous download. It often requires a NASA request. Substituting the classic C-MAPSS subsets is the documented fallback. The loader is written against the classic 26-column text files, not the N-CMAPSS HDF5 schema.

## Airbus helicopter accelerometer set (used)

- What it is: one-minute accelerometer traces at 1024 Hz from Airbus flight-test helicopters, plus a labeled validation file.
- Official page: https://doi.org/10.3929/ethz-b-000415151
- Files: training healthy HDF5 (`dftrain`), validation HDF5 (`dfvalid`), ground-truth CSV, description text.
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. https://creativecommons.org/licenses/by-nc-sa/4.0/
- Rights holder: Airbus SAS. Collected 2018. Released with Garcia, Michau, Ducoffe, Sen Gupta, and Fink, "Time Series to Images," arXiv:2005.07031.
- What the license allows here: non-commercial sharing of the analysis, with attribution, under the same license. The raw HDF5 files are not committed. Derived scores in this repository that come from those files are CC BY-NC-SA. Do not sell them or drop the attribution.
- What it is not: it is not an AH-64 HUMS tape. It is flight-test accelerometers, scaled so absolute levels are meaningless, labeled for abnormal sensor behavior rather than a named mechanical fault.

## What was refused

- Any dataset that asked for a government sponsor, an export check, or a company login.
- Any real tail number, maintenance record, or vibration tape from an operational unit.
