# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.148.1/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT="3.8"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Option] Install Node.js
ARG INSTALL_NODE="true"
ARG NODE_VERSION="lts/*"
RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "source /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY dev_requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/dev_requirements.txt \
   && rm -rf /tmp/pip-tmp

COPY docs/requirements.txt /tmp/pip-tmp/docs/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/docs/requirements.txt \
   && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends pandoc

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1

USER root
RUN python -m pip install --upgrade pip
#RUN pip install -r requirements.txt
#RUN pip install -r docs/requirements.txt


# Make sure the contents of our repo are in ${HOME}.
# These steps are required for use on mybinder.org.
#COPY . ${HOME}
#RUN chown -R ${USER} ${HOME}
# Drop back down to user permissions for things that don't need it.
USER ${USER}