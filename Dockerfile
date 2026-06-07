FROM junwha/ddiff-base:cu12.4.1-py3.10-torch-251214

WORKDIR /workspace

# 1. Initialize venv linked to a specific Torch version
# (Note: This creates a symlink to the base image's venv, thus only one project per torch version is recommended)
RUN bash -c uv_init_torch2.5.1

# 2. Activate venv for uv and python
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"

# 3. Install system dependencies for mpi4py
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    openmpi-bin \
    libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# 4. Install additional packages (without torch)
COPY pyproject.toml /workspace/

RUN uv pip install --no-cache .
