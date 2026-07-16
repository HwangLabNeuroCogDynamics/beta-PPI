# beta-PPI
Code and examples for the paper: Estimating trial-wise modulation of functional connectivity using event-related fMRI 

## Overview

This repository provides analysis code for examining how trial wise variables modulate functional connectivity from a behavioral model. The main analysis implements a beta-series psychophysiological interaction (beta-PPI) framework that quantifies trial-by-trial modulation of functional connectivity by model-derived variables. We performed simulations to test for statistical power, the effects of SNR, and parameter recovery.

The repository includes:

* For the main simulation, check out simul.py.
* For comparison with gPPI, check out gPPI_simul.py
* For an example of how to implement this method for seed-based betaPPI, please see the example.py

## Software 

All analyses were performed using python, with the following tools.

* Python (≥ 3.8)
  Main environment for simulation.

nilearn: 0.12.1

statsmodels: 0.14.6

nibabel:5.3.3 



