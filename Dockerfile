FROM ghcr.io/astral-sh/uv:debian

ENV UV_LINK_MODE=copy
COPY . /project
WORKDIR /project
RUN --mount=type=secret,id=github_token \
    --mount=type=cache,target=/root/.cache/uv \
    uv python install && \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0="url.https://$(cat /run/secrets/github_token)@github.com/.insteadOf" \
    GIT_CONFIG_VALUE_0="https://github.com/" \
    uvx --with PyYAML invoke install --prod