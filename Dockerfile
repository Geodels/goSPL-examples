# Docker image for the goSPL-examples environment.
#
# Builds the `gospl-smoke` conda environment from environment.yml using mamba
# (conda-forge + geodels channels). The image is "environment only": it contains
# the solver stack and JupyterLab but no example notebooks — mount your working
# directory, or build on top of it (see .binder/Dockerfile for the Binder image).
#
# Published to Docker Hub as geodels/gospl-examples by .github/workflows/docker-build.yml.

FROM condaforge/miniforge3:latest

LABEL org.opencontainers.image.title="goSPL-examples" \
      org.opencontainers.image.description="gospl-smoke conda environment built from environment.yml with mamba." \
      org.opencontainers.image.source="https://github.com/Geodels/goSPL-examples" \
      org.opencontainers.image.licenses="GPL-3.0"

# --- Build the conda environment from environment.yml using mamba ---
# Increase download timeouts and retries to handle slow conda-forge mirrors,
# particularly for linux-aarch64 packages on GitHub Actions arm64 runners.
RUN conda config --system --set remote_read_timeout_secs 300 && \
    conda config --system --set remote_connect_timeout_secs 60 && \
    conda config --system --set remote_max_retries 10 && \
    conda config --system --set remote_backoff_factor 5

COPY environment.yml /tmp/environment.yml
# Retry env creation: the conda-forge CDN occasionally drops large linux-aarch64
# packages (e.g. pyarrow-core) mid-download, aborting the whole solve. The package
# cache in /opt/conda/pkgs persists across attempts within this RUN layer, so each
# retry only re-fetches what is still missing and eventually completes.
RUN n=0; \
    until mamba env create -f /tmp/environment.yml; do \
        n=$((n+1)); \
        if [ "$n" -ge 6 ]; then echo "mamba env create failed after $n attempts" >&2; exit 1; fi; \
        echo "mamba env create failed; retry $n/6 in 15s..." >&2; \
        sleep 15; \
        rm -rf /opt/conda/envs/gospl-smoke; \
    done && \
    mamba clean --all --yes && \
    find /opt/conda -follow -type f -name '*.pyc' -delete && \
    rm -f /tmp/environment.yml

# environment.yml declares `name: gospl-smoke`
ENV CONDA_ENV=gospl-smoke
# Put the environment on PATH so python / jupyter / mpirun resolve by default,
# and activate it for interactive shells too.
ENV PATH=/opt/conda/envs/gospl-smoke/bin:$PATH
RUN echo "conda activate gospl-smoke" >> /etc/skel/.bashrc && \
    echo "conda activate gospl-smoke" >> /root/.bashrc

# MPI / libfabric workaround (see README): force TCP transport on hosts where
# OFI auto-selection picks the wrong interface.
ENV FI_PROVIDER=tcp \
    MPICH_CH4_OFI_ENABLE=0

# Open MPI runtime settings for containerized use:
#  - OMPI_MCA_btl=^openib: exclude the InfiniBand BTL outright. There's no RDMA
#    hardware in a container (Codespaces or plain docker run), so without this
#    Open MPI tries to load it anyway and prints a harmless but noisy
#    "mca_base_component_repository_open: unable to open mca_btl_openib:
#    librdmacm.so.1 ... (ignored)" warning on every mpirun.
#  - OMPI_MCA_btl_vader_single_copy_mechanism=none: skip the CMA (cross-memory
#    attach) shared-memory fast path outright. Sandboxed containers (Codespaces
#    in particular) restrict ptrace, so CMA isn't available anyway; without this,
#    Open MPI tries it first and prints a "CMA support is not available due to
#    restrictive ptrace settings" warning before silently falling back.
#  - OMPI_MCA_rmaps_base_oversubscribe=1: don't fail when the requested rank
#    count (`mpirun -np N`) exceeds the slots Open MPI detects. Codespaces
#    machine sizes vary and detected core counts under cgroups can be smaller
#    than expected, so without this, `mpirun -np N` can refuse to launch at all
#    ("not enough slots") — equivalent to always passing --oversubscribe.
#  - OMPI_ALLOW_RUN_AS_ROOT / _CONFIRM: this image (and Codespaces containers
#    generally) run as root with no non-root user configured, so mpirun would
#    otherwise refuse to run without --allow-run-as-root on every invocation.
ENV OMPI_MCA_btl=^openib \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    OMPI_MCA_rmaps_base_oversubscribe=1 \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Threading defaults for the MPI-parallel solver. goSPL distributes work across
# MPI ranks, so each rank must run single-threaded BLAS/OpenMP — otherwise every
# rank spawns as many BLAS threads as there are cores and they oversubscribe the
# CPU, making the container far slower than a native run. Control parallelism via
# `mpirun -n <ranks>` instead. Override at runtime with `docker run -e OMP_NUM_THREADS=N`.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /work
EXPOSE 8888

# Default: launch JupyterLab. Mount your examples with e.g.
#   docker run -it -p 8888:8888 -v "$PWD":/work geodels/gospl-examples:latest
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]