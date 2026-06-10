# goSPL-examples

Series of examples to illustrate the functionalities of goSPL.


**goSPL** (short for ``Global Scalable Paleo Landscape Evolution``) is an open source, GPL-licensed library providing a scalable parallelised Python-based numerical model to simulate landscapes and basins reconstruction at global scale. goSPL is developed by the `EarthCodeLab Group <https://earthcolab.org>`_ at the University of Sydney.

![gospl](https://github.com/Geodels/gospl/blob/master/docs/images/earth.png?raw=true)


[![DOI](https://joss.theoj.org/papers/10.21105/joss.02804/status.svg)](https://doi.org/10.21105/joss.02804)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Geodels/goSPL-examples/HEAD?labpath=Local-examples%2Fstratigraphic_record%2Fmodel_setup.ipynb)

## Launch on Binder

You can try a curated subset of the examples in your browser — no installation required — via [Binder](https://mybinder.org/v2/gh/Geodels/goSPL-examples/HEAD). The Binder image is intentionally limited to two representative examples:

- [`Local-examples/stratigraphic_record`](Local-examples/stratigraphic_record)
- [`Global-examples/continental_flux`](Global-examples/continental_flux)

Binder reuses the prebuilt [`geodels/gospl-examples`](https://hub.docker.com/r/geodels/gospl-examples) Docker image (see [`.binder/Dockerfile`](.binder/Dockerfile)) so the heavy goSPL/PETSc stack does not have to be rebuilt. The full set of examples is best run locally or with Docker (below). Note that Binder sessions are resource-limited, so only the lighter steps of these examples are expected to complete there.

## Installation via Conda

```bash
  mamba env create -f environment.yml
  conda activate gospl-smoke
```

## Installation via Docker

The goSPL image contains all the dependencies and configuration files required to run the examples.

Use the ``gospl:latest`` image to run those examples with the most recent goSPL release.

> **Examples image (built from this repository).** A companion image, [`geodels/gospl-examples`](https://hub.docker.com/r/geodels/gospl-examples), is built automatically from [`environment.yml`](environment.yml) (with `mamba`) by the [`Build and push Docker image`](.github/workflows/docker-build.yml) GitHub Actions workflow and pushed to Docker Hub on each release/tag (or manually). It packages the `gospl-smoke` environment and JupyterLab; mount your examples with `-v "$PWD":/work`. The workflow needs two repository secrets: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

##### Pulling the image

Once you have installed Docker on your system, you can ``pull`` the
[goSPL official image](https://hub.docker.com/u/geodels) as follow::

```bash
  docker pull geodels/gospl:latest
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

## Detailed Conda installation

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
    conda activate gospl-smoke
```

To install other packages, jupyter for example::

```console
    conda install jupyter
```

After your environment has been activated, you can either use VS-code or jupyter for running those examples on your local computer. 

## Repository structure

Each example lives in its own folder and follows the same three-stage workflow:

1. **`build_inputs.ipynb`** / **`model_setup.ipynb`** — build the mesh and the forcing fields (elevation, rainfall, tectonics, sea level) and export them as the `.npz` / `netCDF` files goSPL reads.
2. **`runModel.py`** + **`input-*.yml`** — the goSPL driver script and its YAML configuration; launched from a terminal, optionally under MPI.
3. **`sims-analysis.ipynb`** / **`extract_strata.ipynb`** — post-process the HDF5 outputs (remap to a regular grid; extract elevation, erosion–deposition `erodep`, flow accumulation, stratigraphy) and visualise the results.

Shared post-processing utilities live in [`shared_scripts/`](shared_scripts) (`mapOutputs`, `extractBasin`, `getCatchmentInfo`, `stratal`, `umeshFcts`), and HPC deployment notes are in [`hpc-setup/`](hpc-setup).

### Global examples

| Example | What it demonstrates |
|---|---|
| [`continental_flux`](Global-examples/continental_flux) | Global continental erosion–deposition and sediment flux to the oceans over 1 Myr on a coastline-refined mesh. |
| [`erodep_1My`](Global-examples/erodep_1My) | Global 1 Myr run coupling stream-power erosion, marine deposition and flexural isostasy. |
| [`plate_mvt`](Global-examples/plate_mvt) | Global landscape evolution driven by horizontal plate motion (advection) together with vertical tectonics. |

### Local examples

| Example | What it demonstrates |
|---|---|
| [`generate_2Dmesh`](Local-examples/generate_2Dmesh) | Building 2D unstructured (Voronoi / UGRID) meshes and goSPL input files, including from a GeoTIFF DEM. |
| [`flow_direction`](Local-examples/flow_direction) | Comparison of flow-routing schemes — single-flow (SFD), two-neighbour and multiple-flow-direction (MFD). |
| [`implicit_timestepping`](Local-examples/implicit_timestepping) | Sensitivity of the implicit solver to the time step Δt (500 yr → 5 kyr): stability versus accuracy. |
| [`escarpment_retreat`](Local-examples/escarpment_retreat) | Retreat of a rifted-margin escarpment under fluvial incision, hillslope diffusion, orographic rain and flexure. |
| [`glacial_erosion`](Local-examples/glacial_erosion) | Glacial erosion coupled with river/soil transport and critical-slope diffusion (run scripts only). |
| [`stratigraphic_record`](Local-examples/stratigraphic_record) | Building a passive-margin model that records stratigraphy, then extracting stratal architecture and a Wheeler (chronostratigraphic) chart. |

## Running goSPL-examples

The examples provided here target the ``2026.06.08`` goSPL release (see [`environment.yml`](environment.yml)) and consist of simple local and global models that illustrate the main capabilities of the code. If you are new to goSPL, start with the local example [`stratigraphic_record`](Local-examples/stratigraphic_record) and the global example [`continental_flux`](Global-examples/continental_flux).

A typical run, from an activated environment, builds the inputs in the relevant notebook, then launches the model from a terminal and finally post-processes the outputs in the analysis notebook:

```bash
conda activate gospl-smoke
cd Local-examples/stratigraphic_record
mpirun -np 4 python runModel.py -i input-strati.yml
```

Replace `4` with the number of MPI processes to use and `input-strati.yml` with the example's input file. Add `-v` for verbose output.