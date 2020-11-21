Welcome to HyperFluo, the multidimensional image de-mixer.

This project is under development, but offers a framework for hyperdimensional image processing.

Currently, images are acquired sequentially on the microscope:

First, spectrally resolved images are acquired using Zeiss LSM880. 
These are saved as 32 channel LSM image files. 
Temporally resolved images are acquired using the FLIMBox from ISS - digital frequency domain fluorescence lifetime images. 
These are saved as .fbd files, then converted locally to .R64 files- containing the phase and modulation.

## Dependencies:
* python 3 
* tifffile
`pip install tifffile`

 



