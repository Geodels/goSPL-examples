# goSPL-examples

Series of examples to illustrate the functionalities of goSPL.


**goSPL** (short for ``Global Scalable Paleo Landscape Evolution``) is an open source, GPL-licensed library providing a scalable parallelised Python-based numerical model to simulate landscapes and basins reconstruction at global scale. goSPL is developed by the `EarthCodeLab Group <https://earthcolab.org>`_ at the University of Sydney.

![gospl](https://github.com/Geodels/gospl/blob/master/docs/images/earth.png?raw=true)


[![DOI](https://joss.theoj.org/papers/10.21105/joss.02804/status.svg)](https://doi.org/10.21105/joss.02804)

## Installation via Conda

```bash
  mamba env create -f environment.yml
  conda activate gospl-smoke
```

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


## 🔧 Configure MPI environment variables inside the `gospl-smoke` Conda environment (if necessary)

This is a workaround to prevent OFI polling crashes during `MPI_Finalize()`.
```bash
MPIDI_OFI_handle_cq_error:
OFI poll failed (default nic=utun4: Input/output error)
```
comes from libfabric (OFI) + MPICH automatically selecting the wrong network interface on macOS...

### 1. Activate the environment

```bash
conda activate gospl-smoke
```

This ensures all changes are applied specifically to the `gospl-smoke` environment.

---

### 2. Create activation hook directory

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
```

This directory stores scripts that run automatically whenever the environment is activated.

---

### 3. Define MPI/libfabric environment variables

Create the activation script:

```bash
nano $CONDA_PREFIX/etc/conda/activate.d/mpi.sh
```

Add the following lines:

```bash
export FI_PROVIDER=tcp
export FI_TCP_IFACE=en0
export MPICH_CH4_OFI_ENABLE=0
```

### 💡 What this does:

* `FI_PROVIDER=tcp` → forces libfabric to use TCP instead of OFI auto-selection
* `FI_TCP_IFACE=en0` → forces use of the physical network interface (avoids `utun*`)
* `MPICH_CH4_OFI_ENABLE=0` → disables OFI layer in MPICH CH4

---

### 4. Create deactivation cleanup script (optional but recommended)

```bash
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
nano $CONDA_PREFIX/etc/conda/deactivate.d/mpi.sh
```

Add:

```bash
unset FI_PROVIDER
unset FI_TCP_IFACE
unset MPICH_CH4_OFI_ENABLE
```

### 💡 What this does:

* Cleans up environment variables when leaving the environment
* Prevents leakage into other conda environments or shell sessions

---

### 5. Reload the environment

```bash
conda deactivate
conda activate gospl-smoke
```

This ensures the activation script is executed.

---

### 6. Verify environment variables

```bash
echo $FI_PROVIDER
echo $FI_TCP_IFACE
```

Expected output:

```bash
tcp
en0
```

---

### 7. Test MPI functionality

```bash
mpirun -n 4 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"
```

Expected output:

```
0
1
2
3
```

---

### ✅ Result

At this point, your `gospl-smoke` environment should:

* Avoid `utun*` interfaces
* Use stable TCP-based MPI communication
* Prevent OFI polling crashes during `MPI_Finalize()`
* Work consistently for goSPL/PETSc workflows
