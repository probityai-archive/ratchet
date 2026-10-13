# The no-build demo image: the ratchet tool, a local Atlas store (mongod + mongot) and the scripted
# double-refund scenario. No GPU, no model key, no account, and no network once the image is pulled.
#
#   docker build -f docker/demo.Dockerfile -t ratchet-demo:dev .
#   docker run --rm ratchet-demo:dev ratchet demo double-refund
#
# Every `ratchet` command starts the in-container store on first use and waits until it answers
# search-index calls (about 20 s). The reviewer sequence is in docs/RUNNING.md; docker/demo/check.sh runs it.

ARG ATLAS_IMAGE=mongodb/mongodb-atlas-local:8.3.9-20260928T091948Z

FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

# The store image is RHEL 9 (glibc 2.34) with no package manager; build on the same glibc so every
# wheel uv picks loads there. Every dependency the image installs comes from a registry pinned by hash in
# uv.lock, so the build installs no system package and reads no distribution mirror.
FROM almalinux:9.6 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/ratchet/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /opt/ratchet
RUN uv python install 3.14.8
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY probity-ratchet-cases/ probity-ratchet-cases/
COPY src/ src/
COPY keys/ keys/
RUN uv sync --locked --no-dev --python 3.14.8 \
    && /opt/ratchet/.venv/bin/ratchet --help > /dev/null

FROM ${ATLAS_IMAGE}
USER root
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/ratchet /opt/ratchet
COPY --chmod=755 docker/demo/ratchet-store docker/demo/ratchet /usr/local/bin/
RUN mkdir -p /data/ratchet-demo && chown mongod:mongod /data/ratchet-demo
# Root only until the entrypoint has fixed the hostname and file limit; it then drops to mongod.
USER root
WORKDIR /data/ratchet-demo
# The store listens on loopback only, without auth, inside this container.
# Docker's default seccomp profile refuses user namespaces, so guard children cannot apply the netns
# layer here; seccomp still refuses every socket and Landlock every TCP bind and connect.
ENV RATCHET_MONGODB_URI="mongodb://127.0.0.1:27017/?directConnection=true" \
    RATCHET_DB=ratchet_demo \
    RATCHET_DEMO_DIR=/data/ratchet-demo \
    RATCHET_GUARD_REQUIRE=rlimits,seccomp,landlock \
    RATCHET_VENV=/opt/ratchet/.venv
# Healthy means the store answers search-index calls. It reports unhealthy, and starts nothing, until the
# store is up (the default command starts it; a `docker exec ratchet ...` starts it on first use otherwise).
HEALTHCHECK --interval=5s --timeout=30s --start-period=60s --retries=30 \
    CMD ["/usr/local/bin/ratchet-store", "health"]
ENTRYPOINT ["/usr/local/bin/ratchet-store"]
CMD ["serve"]
