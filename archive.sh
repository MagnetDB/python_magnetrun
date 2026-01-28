#! /bin/bash

set -x # force debug
#set -eo pipefail # add flag to stop when command returns an error

SRCDIR=python_magnetrun
PACKAGE=python-magnetrun

usage(){
   echo ""
   echo "Description:"
   echo "               Builds a debian package"
   echo ""
   echo "Usage:"
   echo "               archive.sh [ <option> ] ... ]"
   echo ""
   echo "Options:"
   echo "-d             Specify the distribution to use"
   echo "-v <x.y.z>     Specify version to build."
   echo ""
   exit 1
}

while getopts "hd:v:" option ; do
   case $option in
       h ) usage ;;
       d ) DIST=$OPTARG ;;
       v ) VERSION=$OPTARG ;;
       ? ) usage ;;
   esac
done
# shift to have the good number of other args
shift $((OPTIND - 1))

# add parameters
: ${VERSION:="0.0.7"}
: ${DIST:="bookworm"}

# cleanup source
find . \( -path "./magnetrun-env/*" -o -path "./pigbrotherdata/*" -o -path "./pigbrother_colddata/*" \) -prune -o -type d -name __pycache__ -print | xargs rm -rf

# create archive

cd ..
tar \
    --exclude-vcs \
    --exclude=feelppdb \
    --exclude=tmp \
    --exclude=debian \
    --exclude=.pc \
    --exclude=.devcontainer \
    --exclude=.vscode \
    --exclude=*.sif \
    --exclude=*.crt \
    --exclude=*.pem \
    --exclude=*log \
    --exclude=*.csv* \
    --exclude=*.orig \
    --exclude=*.new \
    --exclude=*.rej \
    --exclude=python_magnetrun/*.json \
    --exclude=python_magnetrun/*.png \
    --exclude=*env \
    --exclude=data \
    --exclude=srvdata \
    --exclude=pigbrotherdata \
    --exclude=pigbrother_colddata \
    --exclude=outputs \
    --exclude=tmp \
    --exclude=python_magnetrun/*output* \
    --exclude=python_magnetrun/json \
    --exclude=python_magnetrun/png \
    --exclude=*~ \
    --exclude=\#*\# \
    --exclude=pyproject.toml \
    --exclude=poetry.lock \
    -zcvf ${PACKAGE}_${VERSION}.orig.tar.gz ${SRCDIR}

# build package
# disable use of hooks in pbuilder
mkdir -p tmp
cd tmp
cp ../${PACKAGE}_${VERSION}.orig.tar.gz .
tar zxvf ./${PACKAGE}_$VERSION.orig.tar.gz
cp -r ../${SRCDIR}/debian ${SRCDIR}
cd ${SRCDIR}
DIST=${DIST} pdebuild

# clean up
cd ..
rm -rf ${PACKAGE}*
rm -rf ${SRCDIR}

# # upload new package to Lncmi package repository
# cd /var/cache/pbuilder/$DIST-amd64/results
# dupload -t euler_$DIST 

# connect to Lncmi repository server
# update server
