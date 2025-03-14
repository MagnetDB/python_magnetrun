ARG BASE=debian:12

FROM ${BASE}
LABEL maintainer="Christophe Trophime <christophe.trophime@lncmi.cnrs.fr>"

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

# Setup demo environment variables
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    && apt-get -y install git git-lfs iproute2 procps lsb-release \
    #\
    # Install dev tools\
    && apt-get -y install build-essential gfortran \
    && apt-get -y install python3-minimal swig \
    && apt-get -y install python-is-python3 python3-pip python3-jinja2 python3-venv python3-virtualenv \
    && apt-get -y install python3-autopep8 black yapf3 python3-bandit flake8 pydocstyle pylint python3-pytest mypy \
    && apt-get -y install texlive-latex-base texlive-latex-extra dvipng cm-super-minimal

# # Add specific packages \
# RUN apt-get -y install RUN apt-get -y install python3-nlopt python3-tabulate \
#     && apt-get -y install python3-statsmodels \
#     && apt-get -y install python3-matplotlib && apt-get -y install python3-pandas python3-seaborn \
#     && apt-get -y install python3-numpy python3-scipy \
#     && apt-get -y install python3-requests python3-requests-toolbelt \
#     && apt-get -y install python3-natsort python3-rich \
#     && apt-get -y install python3-myst-parser python3-sphinxcontrib-mermaid python3-sphinx-rtd-theme

# # Add specific packages from lncmi debian repository \
# RUN apt install -y debian-keyring \
#     && cp /usr/share/keyrings/debian-maintainers.gpg /etc/apt/trusted.gpg.d \
#     && echo "deb http://euler.lncmig.local/~christophe.trophime@LNCMIG.local/debian/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/lncmi.list \
#     && apt update \
#     && apt-get -y install python3-nptdms python3-freesteam python3-ht python3-pyclaw

# Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.\
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for the non-root user
    && apt-get install -y sudo \
    && if [ "$(lsb_release -cs)" = "focal" ]; then echo "Set disable_coredump false" > /etc/sudo.conf; fi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # add github ssh key
    && mkdir -p ~$USERNAME/.ssh/ \
    && ssh-keyscan github.com >> ~$USERNAME/.ssh/known_hosts \
    && chown -R $USER_UID:$USER_GID ~$USERNAME/.ssh \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y

# To be added:
# python3-myst-parser python3-sphinxcontrib-mermaid python3-sphinx-rtd-theme


# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog


# add motd see https://gist.github.com/ashgillman/a6aa7a5afc9e146a65fb
RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/issue && cat /etc/motd' \
    >> /etc/bash.bashrc \
    ; echo "\
    ===================================================================\n\
    = Python MagnetRun Docker container                               =\n\
    ===================================================================\n\
    \n\
    * To create the virtual env:\n\
    python -m venv ./magnetrun\n\
    * To start the virtual env:\n\
    source ./magnetrun-env/bin/activate\n\
    \n\
    * To quit the virtual env:\n\
    deactivate\n"\

    > /etc/motd

# Switch to USERNAME
USER ${USERNAME}
WORKDIR /home/${USERNAME}
CMD ["bash"]
