# skrf_tests
## Verification of noisy scikit-rf microwave circuit simulation

This repository contains the following folders, which contain Jupyter notebooks using the scikit-rf Python library:

##### amplifier_tests
- Simulations of amplifier noise temperature, how active components affect analysis

##### terminator_temp_tests
- Simulations of terminator temperature experiment. The main goal of the experiment is to see how cooling a terminator
connected to a cavity will effect the final noise temperature. We predict that the noise temperature
spectrum will be flat at room temperature with the room temperature terminator connected, and have a peak at room temperature
falling off to the cooled terminator temperature when the terminator is cooled.

##### cavity_and_circ_tests
- Noise temperature of back of amplifier 
- How leakage and cavity phase shift affects asymmetry in the noise power spectrum


Currently working on cav_and_circ_tests: noise temperature plots of blue circuit
