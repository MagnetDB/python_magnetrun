#! /bin/bash

set +x

# pwd shall contains requirements.txt

VENVDIR=./magnetrun-env
USE_SYSTEM_PACKAGES=1

if [ ! -d $VENVDIR ]; then
   echo "create Python Virtualenv: VENVDIR=${VENVDIR}"
   if [ "$USE_SYSTEM_PACKAGES" == "1" ]; then
      python -m venv --system-site-packages $VENVDIR
   else
      python -m venv $VENVDIR
      . $VENVDIR/bin/activate
      pip install black
      cd python_magnetcooling
      pip install -e .
      cd ..
      deactivate
      
   fi
   . $VENVDIR/bin/activate
   python -m pip install -e .
   deactivate
fi

# add option to properly quit gmsh-env using deactivate
