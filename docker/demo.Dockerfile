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

# probityai (RFC 8785 and DSSE from the jcs-admit and dsse crates) is not on PyPI yet, so its wheel is built
# here from the commit uv.lock pins. The manylinux_2_28 image carries gcc and git and builds against glibc 2.28,
# so the wheel loads on the RHEL 9 store image; Rust comes from rustup, its installer checked against the
# published SHA-256. Only the wheel leaves this stage, and the stage goes when the PyPI pin replaces the source.
FROM quay.io/pypa/manylinux_2_28@sha256:f942fe46374fc45cc597affd63305edabc61b638138a60f75f95198b22419dfa AS native
COPY --from=uv /uv /usr/local/bin/uv
ARG PROBITYAI_COMMIT=58a48b2f33b55cafb72de9f187f0fa53eed74d95
ARG RUSTUP_VERSION=1.28.2
ARG RUST_TOOLCHAIN=1.88.0
RUN arch="$(uname -m)-unknown-linux-gnu" \
    && base="https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${arch}" \
    && curl --proto '=https' --tlsv1.2 -sSfo /tmp/rustup-init "${base}/rustup-init" \
    && echo "$(curl --proto '=https' --tlsv1.2 -sSf "${base}/rustup-init.sha256" | cut -d' ' -f1)  /tmp/rustup-init" | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && rm /tmp/rustup-init
ENV PATH=/root/.cargo/bin:${PATH}
RUN git clone --quiet --filter=blob:none https://github.com/probityai/jcs-admit /src \
    && git -C /src checkout --quiet --detach "${PROBITYAI_COMMIT}" \
    && uv build --wheel --python /opt/python/cp312-cp312/bin/python --out-dir /wheels /src/python

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
COPY --from=native /wheels/ /wheels/
RUN uv sync --locked --no-dev --extra atlas --python 3.14.8 --no-install-package probityai \
    && uv pip install --python /opt/ratchet/.venv/bin/python --no-deps /wheels/probityai-0.1.0-*.whl \
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
