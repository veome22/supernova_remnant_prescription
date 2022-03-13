# supernova_remnant_prescription
Warning: This is a work in progress!

The goal of this project is to set the free parameters in [Mandel, Muller 2020](https://arxiv.org/abs/2006.08360) using observed pulsar velocity distributions.
The posteriors are simulated using a [modified version of COMPAS](https://github.com/veome22/COMPAS/tree/muller_mandel_kick_scatter).

The notebook for calculating model likelihood using pulsar velocity data is [FRONTERA_generate_model_pulsar_prob.ipynb](../main/FRONTERA_generate_model_pulsar_prob.ipynb).

Some useful functions for reading the pulsar data as well as calculating likelihoods are defined in [natal_kick_tools](../main/natal_kick_tools).

The notebook for visualizing the natal kicks produced in COMPAS is [FRONTERA_kick_dist_plots_compas.ipynb](../main/FRONTERA_kick_dist_plots_compas.ipynb).

