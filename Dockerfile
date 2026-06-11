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
RUN mamba env create -f /tmp/environment.yml && \
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

WORKDIR /work
EXPOSE 8888

# Default: launch JupyterLab. Mount your examples with e.g.
#   docker run -it -p 8888:8888 -v "$PWD":/work geodels/gospl-examples:latest
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]