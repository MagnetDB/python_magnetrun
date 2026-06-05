FROM python:3.13-slim

LABEL maintainer="christophe.trophime@lncmi.cnrs.fr"
LABEL description="Minimalist environment for python_magnetrun"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
        libxkbcommon-x11-0 \
        libgl1 \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash magnetrun

WORKDIR /home/magnetrun/app
RUN chown magnetrun:magnetrun /home/magnetrun/app

COPY --chown=magnetrun:magnetrun . .

RUN mkdir -p /mnt/LNCMI_Data/records && \
    chown magnetrun:magnetrun /mnt/LNCMI_Data/records

ENV VIRTUAL_ENV=/home/magnetrun/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV MPLBACKEND=TkAgg

USER magnetrun
ENV HOME=/home/magnetrun

# set alias, OpenMP Threads and OpenBLAS (threads should be 1 to ensure performance)
RUN echo "alias cp='cp -i'" > $HOME/.bash_aliases && \
    echo "alias egrep='egrep --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias fgrep='fgrep --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias grep='grep --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias ls='ls --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias mv='mv -i'" >> $HOME/.bash_aliases && \
    echo "alias rm='rm -i'" >> $HOME/.bash_aliases

RUN pwd && ls -alrth \
    && python3 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir -e "python_magnetcooling[all]" \
    && pip install --no-cache-dir -e ".[all]"

ENTRYPOINT ["/bin/bash"]
