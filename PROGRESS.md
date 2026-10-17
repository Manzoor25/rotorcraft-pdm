# Progress

## Done

- Created the project folder and the Azure resource group `rg-rotorcraft-pdm`.
- Recorded dataset licenses in [DATA_SOURCES.md](DATA_SOURCES.md).
- Downloaded NASA C-MAPSS and the Airbus accelerometer files into `data/raw` (gitignored).
- Deleted the previous bank-marketing online endpoint so it would not keep billing beside this project. The bank model registry was left in place.

## Next

- Finish package install, run tests, and train.
- Deploy the Bicep workspace.
- Write the evaluation report from the real numbers.
- Publish the GitHub repo, the Pages site, and the blog.
- Register models if the storage path allows it, then remove any always-on compute.

## Blockers

- None yet. Storage public access is disabled by subscription policy. That may block a laptop upload. It is recorded in [DECISIONS.md](DECISIONS.md) and will be confirmed when registration is attempted.
