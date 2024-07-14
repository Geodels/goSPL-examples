# goSPL-examples

Series of examples to illustrate the functionalities of goSPL

**Useful links**:
`Binary Installers <https://pypi.org/project/gospl>`__ |
`Source Repository <https://github.com/Geodels/gospl>`__ |
`Issues & Ideas <https://github.com/Geodels/gospl/issues>`__ 

**goSPL** (short for ``Global Scalable Paleo Landscape Evolution``) is an open source, GPL-licensed library providing a scalable parallelised Python-based numerical model to simulate landscapes and basins reconstruction at global scale. goSPL is developed by the `EarthCodeLab Group <https://earthcolab.org>`_ at the University of Sydney.


=========================
Installation via Conda
=========================


One of the simplest way to install not only goSPL, but required dependencies  is with `Anaconda <https://docs.continuum.io/anaconda/>`__, a cross-platform (Linux, Mac OS X, Windows) Python distribution for data analytics and scientific computing.


.. note::

    For **Windows users**, you will need to install Anaconda via the Windows Ubuntu Terminal from WSL. There are several articles on the web to do so (such as this `one <https://emilykauffman.com/blog/install-anaconda-on-wsl>`_)

    For other approaches, some installation instructions for `Anaconda <https://docs.continuum.io/anaconda/>`__ can be found `here <https://docs.continuum.io/anaconda/install.html>`__.

A full list of the packages available as part of the `Anaconda <https://docs.continuum.io/anaconda/>`__ distribution can be found `here <https://docs.continuum.io/anaconda/packages/pkg-docs/>`__.

Another advantage to installing Anaconda is that you don't need admin rights to install it. Anaconda can install in the user's home directory, which makes it trivial to delete Anaconda if you decide (just delete that folder).

After getting Anaconda installed, the user will have already access to some essential Python packages and will be able to install a functioning goSPL environment by following the directives below.


Building goSPL-examples environment
------------------------------------

The next step consists in downloading the conda environment for goSPL. A conda environment is like a virtualenv that allows you to install a specific flavor of Python and set of libraries. For the latest version (`master` branch) of goSPL, this is done by downloading the ``environment.yml`` `file <https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml>`_. To do this you can use the ``curl``::

  curl https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml --output environment.yml

or ``wget`` command::

  wget https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml

This will save the file locally under the same name as it was on github: ``environment.yml``.

Alternatively you can get it from your preferred web browser by clicking on the following link: `environment.yml <https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml>`_ and saving it under the following name ``environment.yml``.

.. note::

  goSPL is not directly packaged as a `Conda <https://conda.pydata.org/docs/>`__ library because some of its dependencies are not available via this installation. The use of the environment file however provides an easy installation approach.

Once the `environment.yml <https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml>`_ file has been downloaded on your system. The following directives provide a step-by-step guide to create a local conda environment for goSPL.

Navigate to the directory containing the `environment.yml <https://raw.githubusercontent.com/Geodels/goSPL-examples/master/environment.yml>`_ file and run the following commands from a terminal window::

    conda env create -f environment.yml

This will create an environment with the dependencies and packages required to run goSPL-examples.

To put your self inside this environment run::

    source activate gospl-environment


To install other packages, jupyter for example::

    conda install jupyter


Running goSPL-examples 
------------------------------------

After your environment has been activated, you can either use VS-code or jupyter for running those examples on your local computer. 