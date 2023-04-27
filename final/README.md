# MSCA_reinforcement_learning final PJ
 
## Prerequisites
- SUMO: You can download SUMO from https://sumo.dlr.de/docs/Installing/index.html. It is required to run SUMO-rl correctly.

## Contents
- experiments/: This contains the codes for running simulations.
- nets/: This contains the net file for SUMO. Refer to the SUMO documentation.
- output/: Save output from the simulation.
- utils/: Contains some utilities for working.

## Warning
Currently, stable baseline cannot be used for training due to the friction between gym and gymnasium. We have to use another package or write our own class for training.