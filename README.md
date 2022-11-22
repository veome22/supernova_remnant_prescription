# supernova_remnant_prescription
This work can now be found as a pre-print at [Kapil V., Mandel I., Berti E., Müller B., 2022](https://arxiv.org/abs/2209.09252). The MNRAS verion is pending.

The goal of this project is to set the free parameters in [Mandel, Muller 2020](https://arxiv.org/abs/2006.08360) model for Neutron Star natal kick velocities, using observed pulsar velocity distributions.
The pulsar observations used to constrain this model come from [Wilcox et. al. 2021](https://iopscience.iop.org/article/10.3847/2041-8213/ac2cc8).

The workflow is the following:
1. A population of stars is evolved using COMPAS, using either the SSE or the BSE mode, with various configurations of v_ns and sigma_ns.

2. From the resulting compact object populations, we identify single pulsars and compute the likelihood of the pulsar observations being drawn from each model in [generate_model_likelihoods.ipynb](../main/generate_model_likelihoods.ipynb).

3. The most likely models are identified in [model_likelihood_analysis_sse.ipynb](../main/model_likelihood_analysis_sse.ipynb) and [model_likelihood_analysis_bse.ipynb](../main/model_likelihood_analysis_bse.ipynb).

4. Pulsar Velocity CDFs of the most likely models are compared to observations in [cdf_plots_sse.ipynb](../main/cdf_plots_sse.ipynb) and [cdf_plots_bse.ipynb](../main/cdf_plots_bse.ipynb).

5. The most likely models are compared to observations using a KS Test in [ks_test.ipynb](../main/ks_test.ipynb).

6. The effect of the most likely kick model on the local BNS merger rate is studied in [COMPAS_PostProcessing.ipynb](../main/COMPAS_PostProcessing.ipynb).

Some useful functions for reading the pulsar data as well as calculating likelihoods are defined in [natal_kick_tools](../main/natal_kick_tools).
