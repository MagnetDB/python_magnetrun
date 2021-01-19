FROM ubuntu:focal

USER root

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# install Feelpp from BinTray Debian repository
RUN apt update && \
    apt install -y lsb-release && \
    echo "deb http://archive.ubuntu.com/ubuntu $(lsb_release -cs)-proposed main restricted"  |  tee -a /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu $(lsb_release -cs)-proposed universe"  |  tee -a /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu $(lsb_release -cs)-proposed multiverse"  |  tee -a /etc/apt/sources.list && \
    apt install -y debian-keyring && \
    cp /usr/share/keyrings/debian-maintainers.gpg /etc/apt/trusted.gpg.d && \
    echo "deb http://euler.GRENOBLE.LNCMI.LOCAL/~trophime/debian/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/lncmi.list && \
    apt update


# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    # \
    # Verify git, process tools, lsb-release (useful for CLI installs) installed\
    && apt-get -y install git iproute2 procps lsb-release \
    #\
    # Install C++ tools\
    && apt-get -y install build-essential python3-minimal libpython3.8 \
    && apt-get -y install python3-nptdms python3-freesteam python3-nlopt python3-tabulate \
    && apt-get -y install python3-statsmodels \
    && apt-get -y install python3-matplotlib texlive-latex-base texlive-latex-extra dvipng \
    && apt-get -y install python3-pandas python3-seaborn \
    && apt-get -y install python3-numpy python3-scipy \
    && apt-get -y install python3-ht python3-pyclaw \
    #\
    # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.\
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for the non-root user
    && apt-get install -y sudo \
    && if [ "$(lsb_release -cs)" = "focal" ]; then echo "Set disable_coredump false" > /etc/sudo.conf; fi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # add github ssh key
    && mkdir ~vscode/.ssh/ \
    && ssh-keyscan github.com >> ~vscode/.ssh/known_hosts \
    && chown -R vscode.$USER_GID ~vscode/.ssh \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Setup demo environment variables
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none

