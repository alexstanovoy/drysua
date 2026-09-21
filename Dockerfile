# Linux/amd64 pins; metadata verified 2026-09-21, not runtime-qualified yet.
FROM docker.io/library/rust:1.98.0-slim-bookworm@sha256:af0579d28b9a7ec5251aaafcb0c0a23dcde5c97065112aae0cc3abeda42d5394 AS rust_toolchain

FROM docker.io/nvidia/cuda:13.3.1-devel-ubuntu26.04@sha256:0ee41c7eac41d2579b268be60db1012ad23b0d4f3222b76566128fe28881a8f8 AS build
COPY --from=rust_toolchain /usr/local/rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/ /opt/rust/
ENV PATH="/opt/rust/bin:${PATH}" \
    CUDA_HOME=/usr/local/cuda \
    CC=/usr/bin/gcc-15 \
    CXX=/usr/bin/g++-15 \
    NVCC_CCBIN=/usr/bin/g++-15 \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc-15
# Trusted distribution repositories; record resolved versions, not bit reproducibility.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates g++-15 python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY bota/ bota/
COPY drysua/ drysua/
ARG CUDA_COMPUTE_CAP
RUN test -n "$CUDA_COMPUTE_CAP" && \
    python3 /src/drysua/docker/source_identity.py && \
    . /opt/drysua/source.env && \
    export CUDA_COMPUTE_CAP && \
    cargo build --manifest-path /src/drysua/Cargo.toml --locked --release --quiet \
      --no-default-features --features builtin,cuda --bin drysua && \
    binary_hash="$(sha256sum /src/drysua/target/release/drysua)" && \
    export DRYSUA_BINARY_SHA256="${binary_hash%% *}" && \
    python3 -c 'import json,os,pathlib; p=pathlib.Path; source=p("/opt/drysua/source.sha256").read_text().strip(); p("/opt/drysua/provenance.json").write_text(json.dumps({"source_sha256":source,"binary_sha256":os.environ["DRYSUA_BINARY_SHA256"],"rust":"1.98.0","cuda":"13.3.1","cuda_compute_cap":os.environ["CUDA_COMPUTE_CAP"],"features":"builtin,cuda","scope":"fresh source build, not frozen B40"})+"\n")'

FROM docker.io/nvidia/cuda:13.3.1-runtime-ubuntu26.04@sha256:b9321b748007329ae6a63261eb041612d18b802e23a485717cfa3584d640dd57 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/drysua/target/release/drysua /usr/local/bin/drysua
COPY --from=build /opt/drysua/ /opt/drysua/
COPY drysua/docker/runtime_guard.py /opt/drysua/runtime_guard.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 1000:1000
WORKDIR /run-data
ENTRYPOINT ["python3", "/opt/drysua/runtime_guard.py"]
