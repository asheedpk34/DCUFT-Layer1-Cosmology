# DCUFT Layer-1 Cosmology

This repository contains the Python code and data used for the **Dynamical Charged Cosmic Unifying Φ-Field Theory (DCUFT)**, also known as **Layer-1 Cosmology**. The analysis reproduces the results from:

**Asheed Mohamed, "Layer-1 Cosmology: Testable Predictions from the DCUFT Framework"**

---

## Overview

DCUFT is an extension of the standard ΛCDM cosmology that introduces a hypothesized "information layer" (Layer-1). This framework modifies the expansion history and the growth of cosmic structure. This repository includes:

- MCMC sampling of cosmological parameters (`H0`, `Ωm`, `log10Λ`)
- Fisher matrix analysis for parameter uncertainties
- Plot generation for:
  - Hubble parameter `H(z)`
  - Growth function `fσ8(z)`
  - Posterior corner plots

---

## Folder Structure
DCUFT-Layer1/

data/

growth_data.txt

Pantheon-SHOES.dat.txt

sdss_DR12Consensus_final.txt

# Observational data files

BAO_consensus_covtot_dM.txt

scripts/

Mcmc_code.py

plotting.py

|

results/

# Python

#

scripts for analysis

Fisher_matrix.py |

Output plots and MCMC chains (created after running scripts)

corner_plot.png

fs8_plot.png

.gitignore

Hubble_plot.png

README.md

#This file

LICENSE
