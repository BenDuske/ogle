# Ogle — ML-lineage drift agent for DataHub.
#
# Builds a lean image that can walk a live DataHub GMS
# (`ogle check --gms http://datahub-gms:8080 --discover`). The live walk needs
# the `datahub` extra (acryl-datahub); the pure offline pipeline
# (signatures in → drift out) needs no extra deps. Toggle via OGLE_EXTRAS:
#   docker build -t ogle:dev .                       # live-capable (default)
#   docker build --build-arg OGLE_EXTRAS= -t ogle .  # core only, fastest build
FROM python:3.12-slim

ARG OGLE_EXTRAS=datahub

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package. Copy only what the build needs so edits to tests/docs
# don't bust this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN if [ -n "$OGLE_EXTRAS" ]; then \
        pip install ".[$OGLE_EXTRAS]"; \
    else \
        pip install .; \
    fi

# Unprivileged runtime. /data holds the baseline store (mount it to persist
# across runs — the drift memory that makes Ogle page once, not every sweep).
RUN useradd --create-home --uid 10001 ogle \
    && mkdir -p /data \
    && chown -R ogle:ogle /data
USER ogle
VOLUME ["/data"]

# `ogle` is a CLI, not a daemon: the entrypoint is the tool, the default
# command prints help. docker-compose overrides CMD with a real `check` sweep.
ENTRYPOINT ["ogle"]
CMD ["--help"]
