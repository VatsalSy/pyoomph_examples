# Upstream-motion-of-oil-droplets-in-co--axial-Ouzo-flow-due-to-Marangoni-forces
This repository contains the codes used to reproduce the simulations reported in the manuscript “Upstream motion of oil droplets in co-axial Ouzo flow due to Marangoni forces” ([link](https://doi.org/10.1039/D5SM00848D)). The paper reports on experiments with a co-flow configuration where oil droplets are formed at a certain distance from the nozzle due to the Ouzo effect and move upstream due to Marangoni forces arising from concentration gradients in the surrounding phase. The code in this repository allows to qualitatively reproduce the observed phenomena, by computing the required flow rate of the inner nozzle to maintain a stationary droplet at a given distance from the nozzle.

Citation:
```
@article{bisswanger2026upstream,
  title={Upstream motion of oil droplets in co-axial Ouzo flow due to Marangoni forces},
  author={Bisswanger, Steffen and Rocha, Duarte and Dehe, Sebastian and Diddens, Christian and Baier, Tobias and Lohse, Detlef and Hardt, Steffen},
  journal={Soft matter},
  volume={22},
  number={3},
  pages={567--577},
  year={2026},
  publisher={Royal Society of Chemistry}
}
```

## Prerequisites
Assuming you have installed Python 3.9 to 3.13, you can install pyoomph via:
  ```
  python -m pip install pyoomph
  ```
On Apple Silicon (M1–M4) run the above in a Rosetta 2 terminal. If installation fails, consult the pyoomph installation guide: https://pyoomph.readthedocs.io/en/latest/tutorial/installation.html. For persistent issues, contact the authors.

## Introduction
We numerically evaluate the quasi-stationary solutions that maintain an oil droplet at a fixed distance from the nozzle in the co-axial flow configuration. The simulations are a simplified representation of the experimental setup, where we model a pure anethole droplet subject to drag and Marangoni forces exerted by the surrounding flow of water and ethanol. The geometry is adjusted to match the experimental conditions. We adjust the inner jet (ethanol) flow rate in order to nullify the force balance on the droplet.

## Guide through scripts 
The repository contains the following scripts:
- `jet_profile.py`: Quasi-stationary simulation of the jet profile without droplets, based on the model in Sec 3.2, reproduces the results shown in Fig. 5 of the manuscript. Here, we consider the inner jet to be a mixture of ethanol (88%) and anethole (12%) and the outer jet to be pure water, as in the experiments. The results show the zones where the Ouzo effect is expected to occur.
- `Qjet_at_fix_z.py`: Quasi-stationary simulation to compute the required inner jet flow rate to maintain a droplet at a fixed distance from the nozzle, based on the model in Sec 3.3, reproduces the results shown in Fig. 6 of the manuscript. Here, we consider a pure anethole droplet, the inner jet to be pure ethanol and the outer jet to be pure water.

## Usage
To run the simulations, navigate to the directory containing the desired script and execute it using Python. For example:
```
python3 jet_profile.py
```
This will run the jet profile simulation without droplets as described in the manuscript.
Results will be saved in the output folder with the same name as the script.