# Image for the hessian.AI 42 cluster (MLDE / Determined 0.35.0).
#
# Layered on the image the cluster's own config_blueprint.yaml ships, NOT built from a bare
# CUDA base. That base already carries a matching CUDA/PyTorch stack and the Determined 0.35.0
# harness, so this only has to add the packages our experiments need on top. It also removes
# any risk of a harness/master version mismatch.
#
# (The onboarding slides show py-3.8-pytorch-1.12; that is stale — the live blueprint uses the
# NGC image below.)
#
# This is the reproducible route. The faster route to a first result is to skip Docker
# entirely and `pip install --target` the extras onto PFSS — see cluster/README.md.
#
# Build on amd64 (the cluster) — cross-building from Apple Silicon works but is slow:
#   docker buildx build --platform linux/amd64 \
#     -f cluster/Dockerfile -t <REGISTRY>/rm-bias:latest --push .

FROM determinedai/pytorch-ngc:0.35.0

ENV PIP_NO_CACHE_DIR=1

WORKDIR /opt/app

# requirements-analysis.txt pulls in requirements.txt via `-r`, so this covers the base stack
# (transformers, accelerate, datasets, textstat) plus scikit-learn, concept-erasure (LEACE)
# and markdown-it-py.
#
# torch is deliberately left to the base image: the NGC build is tuned for this hardware and
# reinstalling it from PyPI would replace a matched CUDA build with a generic one.
COPY requirements.txt requirements-analysis.txt ./
RUN sed -i '/^torch/d' requirements.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements-analysis.txt

# Fail the build rather than the job if the stack is not actually importable.
RUN python -c "import torch, transformers, sklearn, concept_erasure, textstat; \
print('torch', torch.__version__, '| transformers', transformers.__version__); \
assert torch.__version__ >= '2.1', 'base image torch too old for these reward models'"

# Code and data are NOT copied in: they live on PFSS and are bind-mounted at run time
# (cluster/config.yaml), so a code change never requires an image rebuild.
