# goSPL-examples

Series of examples to illustrate the functionalities of goSPL.


**goSPL** (short for ``Global Scalable Paleo Landscape Evolution``) is an open source, GPL-licensed library providing a scalable parallelised Python-based numerical model to simulate landscapes and basins reconstruction at global scale. goSPL is developed by the `EarthCodeLab Group <https://earthcolab.org>`_ at the University of Sydney.

![gospl](https://github.com/Geodels/gospl/blob/master/docs/images/earth.png?raw=true)


[![DOI](https://joss.theoj.org/papers/10.21105/joss.02804/status.svg)](https://doi.org/10.21105/joss.02804)

## Installation via Docker

The goSPL image contains all the dependencies and configuration files required to run the examples.

The ``gospl:2024.09.01`` is required for running those examples.

##### Pulling the image

Once you have installed Docker on your system, you can ``pull`` the
[goSPL official image](https://hub.docker.com/u/geodels) as follow::

```bash
  docker pull geodels/gospl:2024.09.01
```
##### Starting the container from a terminal

You can then start a docker container (an instance of
an image)::

```bash
  docker run -it -p 8888:8888 -d -v localDIR:/notebooks
```
where `localDIR` is the directory that contains the examples folder `goSPL-examples`.

Once Docker is running, you could open the Jupyter notebooks on a web browser at the following address: `http://localhost:8888 <http://localhost:8888>`_. Going into the `/notebooks` folder you will access your ``localDIR`` directory.

To run goSPL, you will need to use the terminal from the Jupyter interface. To activate the goSPL environment where all the libraries are installed you will have to run the following command:
```bash
  conda activate gospl
```

Depending on your operating system, you will be able to configure the docker application to set your resources: CPUs, memory, swap, or Disk image size. This will improve the performance of the run.

> Note that you could use the Dashboard from Docker instead of passing through the terminal to download the goSPL Docker image.

## Installation via Conda

One of the simplest way to install not only goSPL, but required dependencies  is with [Anaconda](https://docs.continuum.io/anaconda/), a cross-platform (Linux, Mac OS X, Windows) Python distribution for data analytics and scientific computing.

> For **Windows users**, you will need to install Anaconda via the Windows Ubuntu Terminal from WSL. There are several articles on the web to do so (such as this [one](https://emilykauffman.com/blog/install-anaconda-on-wsl))

A full list of the packages available as part of the [Anaconda](https://docs.continuum.io/anaconda/) distribution can be found [here](https://docs.continuum.io/anaconda/packages/pkg-docs/).

Another advantage to installing Anaconda is that you don't need admin rights to install it. Anaconda can install in the user's home directory, which makes it trivial to delete Anaconda if you decide (just delete that folder).

After getting Anaconda installed, the user will have already access to some essential Python packages and will be able to install a functioning goSPL environment by following the directives below.

### Building goSPL-examples environment

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

After your environment has been activated, you can either use VS-code or jupyter for running those examples on your local computer. 

## Running goSPL-examples 

The series of examples provided here are related to the ``2024.09.01`` goSPL branch and consist in simple local and global models that illustrate the main capabilities of the code. You might want to start with the local example called `stratigraphic_record` and the global example called `continental_flux`.
