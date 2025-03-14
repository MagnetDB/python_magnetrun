#! /bin/bash

if [ ! -d $HOME/gmsh-env ]; then
    python -m venv --system-site-packages $HOME/gmsh-env &&\
    . $HOME/gmsh-env/bin/activate && \
    python -m pip install gmsh
fi

# add option to properly quit gmsh-env using deactivate