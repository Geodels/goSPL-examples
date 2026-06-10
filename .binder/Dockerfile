# Binder image for goSPL-examples.
#
# Reuses the prebuilt environment image (built from environment.yml by
# .github/workflows/docker-build.yml) so Binder does NOT rebuild the heavy
# goSPL / PETSc / GMT stack — which would exceed mybinder.org build limits.
#
# NB: this requires geodels/gospl-examples:latest to exist on Docker Hub first
# (publish a release or run the "Build and push Docker image" workflow).
#
# repo2docker builds this file with the repository root as the build context,
# so the COPY paths below are relative to the repo root.

FROM geodels/gospl-examples:latest

# --- repo2docker / Binder requirements: a non-root user with UID 1000 ---
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV NB_USER=${NB_USER} \
    NB_UID=${NB_UID} \
    HOME=/home/${NB_USER}

USER root
RUN useradd --create-home --shell /bin/bash --uid ${NB_UID} ${NB_USER}

# Binder launches the server via `jupyterhub-singleuser`; add it (plus the
# classic notebook server) to the environment. JupyterLab is already present.
RUN mamba install -n gospl-smoke -y -c conda-forge jupyterhub-singleuser notebook \
    && mamba clean --all --yes

# --- Copy ONLY the two requested examples, preserving the repo layout ---
# The notebooks do `from scripts import ...`; that package's __init__.py walks up
# three parent directories to the repo root and re-exports from shared_scripts/,
# so Global-examples/, Local-examples/ and shared_scripts/ must keep this layout.
COPY Global-examples/continental_flux     ${HOME}/Global-examples/continental_flux
COPY Local-examples/stratigraphic_record  ${HOME}/Local-examples/stratigraphic_record
COPY shared_scripts                       ${HOME}/shared_scripts
COPY README.md                            ${HOME}/README.md

RUN chown -R ${NB_UID}:${NB_UID} ${HOME}

USER ${NB_USER}
WORKDIR ${HOME}
