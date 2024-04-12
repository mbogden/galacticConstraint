# GalacticConstraints

This repository, **GalacticConstraints**, is part of an ongoing effort to reorganize and streamline the development of a scientific software project aimed at exploring and analyzing galactic constraints. The reorganization focuses on improving code structure, enhancing readability, and facilitating easier collaboration and maintenance.   The old repository can be found [here](https://github.com/mbogden/galaxyJSPAM.git).


# Table of Contents
- [Status](#status)
- [Installation](#installation)
- [Instructions](#Instructions)
- [Overview](#overview)
- [References](#references)

# Status<a id ="status">

## TO DO List:
This list is primary intended for my use (Matthew Ogden). 
- [] Update docker image to Python 3.11.


# Installation<a id="installation">

## System Pre-requisites
GalacticConstrains is reliant on the below packages and assumes you are using Python 3.11.
```
sudo apt install mpich
sudo apt install libgl1-mesa-glx
python3.11 -m pip install --upgrade pip
``` 

For our cluster, MPI is an optional package and can be loaded in the following way.  Must be called before installation of mpi4py and running. 
```
module load gnu12
module load openmpi4
module load py3-mpi4py
```

## Git Repository 

Currently, active work is being done in a student's repository on GitHub and can be downloaded via git command.  Move into the directory and run the Makefile to install. 
```
git clone https://github.com/mbogden/galacticConstraints.git
cd galacticConstraints
make
```

## Python package install
The default setup uses a virtual environment to initialize and run the programs. The following commands will create a new virtual environment name "venv_gal", and install needed python packages. NOTE: The name "venv_gal" is arbitrary and you can name it however you please.

```
python3.11 -m venv venv_gal
source venv_gal/bin/activate
python3.11 -m pip install -r requirements.txt
```
 
## DOCKER IMAGE (BETA)
Currently, a docker image are being created and tested to work with galacticConstraints project.  If you are familiar with docker images then you may try to use the following.  Our current image is under construction and can be found in a DockerHub [repo](https://hub.docker.com/repository/docker/ogdenm12/beta-2/general). This docker image may or may not work for you and is not fully supported yet.


# Quickstart<a id="Instructions"> 
Assuming you have completed the installation instructions above, this is a quick guide to how to start the code. 

## Activate environment  
Launch the virtual environment before you run code.     
```
source venv_gal/bin/activate 
```

### Docker + Singularity
Alternatively, you can use a docker image with Singularity.  Assuming you are a Babbage node at MTSU, you can run the following commands to run the docker image. 
```
module load singularity

# For Command Line Interface
singularity shell /home/mbo2d/images/babbage-beta-2.sif

# For Jupyter Notebook
singularity exec  --nv /home/mbo2d/images/babbage-beta-2.sif jupyter lab --no-browser --port=7608 
```


# Current Status
## Status<a id="status">
As of now, this repository is in the early stages of setup and development. The codebase is being restructured to adopt better software development practices, including modular design, comprehensive documentation, and robust testing frameworks.

# References
- Holincheck, A. J., Wallin, J. F., Borne, K., Fortson, L., Lintott, C., Smith, A. M., Bamford, S., Keel, W. C., & Parrish, M. (2016). Galaxy Zoo: Mergers – Dynamical models of interacting galaxies. Monthly Notices of the Royal Astronomical Society, 459(1), 720–745. https://doi.org/10.1093/MNRAS/STW649

- Wallin, J. F., Holincheck, A. J., & Harvey, A. (2016). JSPAM: A restricted three-body code for simulating interacting galaxies. Astronomy and Computing, 16, 26–33. https://doi.org/10.1016/J.ASCOM.2016.03.005
- 

# Acknowledgments
This project incorporates work from [Dr. Wallin](https://github.com/jfwallin/JSPAM), used under [Academic Free License v3.0](https://opensource.org/license/afl-3-0-php). We extend our gratitude to Dr. Wallin for their contributions to the community and for granting permission to utilize and adapt their work.
