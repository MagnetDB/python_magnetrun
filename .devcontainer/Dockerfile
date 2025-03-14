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
    && apt install -y lsb-release \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    # \
    # Verify git, process tools, lsb-release (useful for CLI installs) installed\
    && apt-get -y install git iproute2 procps lsb-release

# Install python tools\
RUN apt-get -y install build-essential python3-minimal python-is-python3 python3-venv python3-virtualenv \
    && apt-get -y install python3-nlopt python3-tabulate \
    && apt-get -y install python3-statsmodels \
    && apt-get -y install python3-matplotlib texlive-latex-base texlive-latex-extra dvipng \
    && apt-get -y install python3-pandas python3-seaborn \
    && apt-get -y install python3-numpy python3-scipy \
    && apt-get -y install python3-rich \
    && apt-get -y install python3-iapws

# # install specific python packages from lncmi repo
# RUN apt install -y debian-keyring && \
#     cp /usr/share/keyrings/debian-maintainers.gpg /etc/apt/trusted.gpg.d && \
#     echo "deb http://euler.lncmig.local/~christophe.trophime@LNCMIG.local/debian/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/lncmi.list && \
#     apt update && \
#     apt-get -y install python3-nptdms && \
#     apt-get -y install python3-pyclaw && \  
#     apt-get -y install python3-ht && \
#     apt-get -y install python3-freesteam
    
# Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.\
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for the non-root user
    && apt-get install -y sudo \
    && if [ "$(lsb_release -cs)" = "focal" ]; then echo "Set disable_coredump false" > /etc/sudo.conf; fi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME

# add github ssh key
RUN mkdir -p ~$USERNAME/.ssh/ \
    && ssh-keyscan github.com >> ~$USERNAME/.ssh/known_hosts \
    && chown -R ${USER_UID}:$USER_GID ~$USERNAME/.ssh \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y 

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog


# add motd see https://gist.github.com/ashgillman/a6aa7a5afc9e146a65fb
RUN apt install tini \
    && echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/issue && cat /etc/motd' \
    >> /etc/bash.bashrc \
    ; echo "\
===================================================================\n\
= Gmsh Docker container                                        =\n\
===================================================================\n\
\n\
* To start the virtual env and run Gmsh:\n\
source $HOME/gmsh-env/bin/activate\n\
gmsh\n\
\n\
* To quit the virtual env:\n\
deactivate\n"\
    > /etc/motd

# Switch to USERNAME
USER ${USERNAME}
RUN pwd && \
    ls -lrth

COPY start-venv.sh /home/${USERNAME}
RUN /home/${USERNAME}/start-venv.sh

WORKDIR /home/${USERNAME}
ENTRYPOINT ["tini", "--"]
CMD ["bash"]
