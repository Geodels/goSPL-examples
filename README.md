# goSPL-examples

Series of examples to illustrate the functionalities of goSPL.


**goSPL** (short for ``Global Scalable Paleo Landscape Evolution``) is an open source, GPL-licensed library providing a scalable parallelised Python-based numerical model to simulate landscapes and basins reconstruction at global scale. goSPL is developed by the `EarthCodeLab Group <https://earthcolab.org>`_ at the University of Sydney.

|    |    |
| --- | --- |
| Build Status | [![Coverage Status](https://coveralls.io/repos/github/Geodels/gospl/badge.svg?branch=master)](https://coveralls.io/github/Geodels/gospl?branch=master)  [![DOI](https://zenodo.org/badge/206898115.svg)](https://zenodo.org/badge/latestdoi/206898115) |
| Latest release | [![Github release](https://img.shields.io/github/release/Geodels/gospl.svg?label=tag&colorB=11ccbb)](https://github.com/Geodels/gospl/releases) [![PyPI version](https://badge.fury.io/py/gospl.svg?colorB=cc77dd)](https://pypi.org/project/gospl) |
| Features | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)|


![gospl](https://github.com/Geodels/gospl/blob/master/docs/images/earth.png?raw=true)


[![DOI](https://joss.theoj.org/papers/10.21105/joss.02804/status.svg)](https://doi.org/10.21105/joss.02804)

## Installation via Docker




## Installation via Conda

One of the simplest way to install not only goSPL, but required dependencies  is with [Anaconda](https://docs.continuum.io/anaconda/), a cross-platform (Linux, Mac OS X, Windows) Python distribution for data analytics and scientific computing.

> For **Windows users**, you will need to install Anaconda via the Windows Ubuntu Terminal from WSL. There are several articles on the web to do so (such as this [one](https://emilykauffman.com/blog/install-anaconda-on-wsl))

A full list of the packages available as part of the [Anaconda](https://docs.continuum.io/anaconda/) distribution can be found [here](https://docs.continuum.io/anaconda/packages/pkg-docs/).

Another advantage to installing Anaconda is that you don't need admin rights to install it. Anaconda can install in the user's home directory, which makes it trivial to delete Anaconda if you decide (just delete that folder).

After getting Anaconda installed, the user will have already access to some essential Python packages and will be able to install a functioning goSPL environment by following the directives below.


Building goSPL-examples environment
------------------------------------

The next step consists in downloading the conda environment for goSPL. A conda environment is like a virtualenv that allows you to install a specific flavor of Python and set of libraries. For the latest version (`master` branch) of goSPL, this is done by downloading the ``environment.yml`` `file <https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml>`_. To do this you can use the ``curl``:

```console
  curl https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml --output environment.yml
```

or ``wget`` command:

```console
  wget https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml
```

This will save the file locally under the same name as it was on github: ``environment.yml``.

Alternatively you can get it from your preferred web browser by clicking on the following link: [environment.yml](https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml) and saving it under the following name ``environment.yml``.

>  goSPL is not directly packaged as a [Conda](https://conda.pydata.org/docs/) library because some of its dependencies are not available via this installation. The use of the environment file however provides an easy installation approach.

Once the [environment.yml](https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml) file has been downloaded on your system. The following directives provide a step-by-step guide to create a local conda environment for goSPL.

Navigate to the directory containing the [environment.yml](https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml) file and run the following commands from a terminal window:

```console
    conda env create -f environment.yml
```

This will create an environment with the dependencies and packages required to run goSPL-examples.

To put your self inside this environment run::


```console
    conda activate gospl
```

To install other packages, jupyter for example::

```console
    conda install jupyter
```

Running goSPL-examples 
------------------------------------

### Anaconda install 

After your environment has been activated, you can either use VS-code or jupyter for running those examples on your local computer. 