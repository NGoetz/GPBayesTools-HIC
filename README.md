This project is a fork of https://github.com/Hendrik1704/GPBayesTools-HIC
It is shipped with additional submission scripts for running emulator generation, validation and posterior generation on a  slurm cluster.
For training, use 'submit_all.sh'. For each training set, the workflow will be executed. The relevant data is parsed from the name of the training set.

For plotting the emulator validation, execute 'read_multiple_emulator_errors_files' in 'EmulatorValidationPlot.ipynb' with the correct input data name and number of validation points. For 'ClosureTest.ipynb', 'Prediction.ipynb', 'Energies.ipynb', 'Rapidity.ipynb' and 'SensitivityAnalysis.ipynb', you have to choose the right path to the emulator and, except for 'SensitivityAnalysis.ipynb', the MCMC file. They have to match in dimensions, of course. All of these need also the path to the pickled experimental data, and an input base config, from which we parse the value range of the parameters of the prior.


Original README
# GPBayesTools-HIC

Gaussian Process Bayesian Toolkit with Monte Carlo Sampler Integration for Heavy Ion Collisions

This toolkit implements a wrapper for Gaussian Process (GP) emulators and Monte Carlo (MC) samplers used in 
high-energy heavy-ion simulations.

The following wrappers for GP emulators are currently included:
- Scikit Learn GP emulator wrapper
- PCGP and PCSK wrapper for the GPs implemented in the [surmise](https://github.com/bandframework/surmise) package of the [BAND](https://bandframework.github.io/) Collaboration

The following wrappers for MC sampling are included:
- MCMC wrapper for the [emcee](https://github.com/topics/emcee) package
- [PTMCMC](https://github.com/willvousden/ptemcee) (Parallel Tempering Markov Chain Monte Carlo) wrapper
    - Not recommend to use this one for larger runs. There are problems with the parallelization.
- [PTLMC](https://github.com/bandframework/surmise) from the surmise package (Parallel Tempering Langevin Monte Carlo)
- [pocoMC](https://github.com/minaskar/pocomc) Preconditioned Monte Carlo method for accelerated Bayesian inference


:exclamation: The jupyter notebooks are just meant as examples for how to use the emulators and samplers and analyze the output.
Paths and data files need the proper input formats.