# discontinuous SWSSB transition

This repository contains codes and data for our study of discontinuous strong-to-weak spontaneous symmetry breaking (SWSSB) transitions in open quantum systems.

---

## Scripts

The `scripts/` directory contains Julia scripts to reproduce data used in the paper.

### Transverse-field Ising model under dephasing

- **`Ising_fidelity_random_initial_state.jl`** : Compute time evolution of the fidelity correlator for random initial states.
- **`Ising_renyi_random_initial_state.jl`** : Compute time evolution of the R\'enyi-2 correlator for random initial states.
- **`Ising_renyi_thermal_initial_state.jl`** : Compute time evolution of the R\'enyi-2 correlator for thermal or many-body localized initial states.
- **`Ising_renyi_ghz_initial_state.jl`** : Compute time evolution of the R\'enyi-2 correlator for the GHZ initial state.
- **`Ising_renyi_Dicke_initial_state.jl`** : Compute time evolution of the R\'enyi-2 correlator for the Dicke initial states.
- **`Ising_renyi_rainbow_initial_state.jl`** : Compute time evolution of the R\'enyi-2 correlator for the rainbow initial states.
- **`Ising_entropy_random_initial_state.jl`** : Compute time evolution of the global entropy for random initial states.
- **`Ising_renyi_random_initial_state.jl`** : Compute time evolution of the R\'enyi-2 correlator for random initial states.
- **`Ising_renyi_cluster_mean_field.jl`** : Compute time evolution of the R\'enyi-2 correlator for random initial states by using the cluster mean-field approximation.

### Hard-core boson model under dephasing

- **`HCB_fidelity_random_initial_state.jl`** : Compute time evolution of the fidelity correlator for random initial states.
- **`HCB_renyi_random_initial_state.jl`** : Compute time evolution of the R\'enyi-2 correlator for random initial states.

----

## Data

The `data/` directory contains data used in the paper.

- **`Ising_fidelity_renyi_random_initial_state_plot.ipynb`** : Plot time evolution of the fidelity and R\'enyi-2 correlators for the Ising model with random initial states (Fig. 2 in the main paper).
- **`renyi_thermal_MBL_initial_state_plot.ipynb`** : Plot time evolution of the R\'enyi-2 correlator for the Ising model with thermal and many-body localized initial states (Fig. 3 in the main paper).
- **`entropy_random_initial_state_plot.ipynb`** : Plot time evolution of the global entropy for the Ising model with random initial states (Fig. 4 in the main paper).
- **`HCB_fidelity_renyi_random_initial_state_plot.ipynb`** : Plot time evolution of the fidelity and R\'enyi-2 correlators for the hard-core boson model with random initial states (Fig. S3 in Supplemental Material).
- **`renyi_GHZ_Dicke_rainbow_initial_state_plot.ipynb`** : Plot time evolution of the R\'enyi-2 correlator for the Ising model with the GHZ, Dicke, and rainbow initial states (Figs. S4-S6 in Supplemental Material).
