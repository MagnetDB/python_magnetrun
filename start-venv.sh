#! /bin/bash

set +x

# pwd shall contains requirements.txt

usage() {
    echo "Usage: $0 [-d VENVDIR] [-s] [-p pkg:extras] [-h]"
    echo ""
    echo "Options:"
    echo "  -d VENVDIR   Path to the virtual environment directory (default: ./magnetrun-env)"
    echo "  -s           Disable system site-packages (enabled by default)"
    echo "  -p pkg:extras  Add a prerequisite local package to install before the main one"
    echo "               Format: 'directory' or 'directory:[extras]'"
    echo "               Can be repeated for multiple packages"
    echo "  -h           Show this help message"
    exit 0
}

VENVDIR=./magnetrun-env
USE_SYSTEM_PACKAGES=1
# Local packages to install (with extras) before the main package
# Format: "directory:extras" or "directory" if no extras
# Example with multiple packages:
#   PREREQ_PACKAGES=(
#       "python_magnetcooling:[fitting]"
#       "python_magnettools"
#       "python_magnetgeo:[meshing,export]"
#   )
# Equivalent command line:
#   ./start-venv.sh -p "python_magnetcooling:[fitting]" -p "python_magnettools" -p "python_magnetgeo:[meshing,export]"
PREREQ_PACKAGES=(
    "python_magnetcooling:[fitting]"
)

while getopts "d:sp:h" opt; do
    case $opt in
        d) VENVDIR="$OPTARG" ;;
        s) USE_SYSTEM_PACKAGES=0 ;;
        p) PREREQ_PACKAGES+=("$OPTARG") ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ ! -d $VENVDIR ]; then
   echo "create Python Virtualenv: VENVDIR=${VENVDIR}"
   if [ "$USE_SYSTEM_PACKAGES" == "1" ]; then
      python -m venv --system-site-packages $VENVDIR
   else
      python -m venv $VENVDIR
    fi

    cleanup() {
        echo "Error: virtualenv setup failed. Removing ${VENVDIR} to prevent a broken environment."
        deactivate 2>/dev/null
        rm -rf "$VENVDIR"
        exit 1
    }
    trap cleanup ERR

    . $VENVDIR/bin/activate
    pip install black
    for entry in "${PREREQ_PACKAGES[@]}"; do
        pkg="${entry%%:*}"
        extras="${entry#*:}"
        (
            cd "$pkg" || exit 1
            if [ "$extras" != "$pkg" ] && [ -n "$extras" ]; then
                pip install -e ".$extras"
            else
                pip install -e .
            fi
        )
    done
    python -m pip install -e .
    deactivate
    trap - ERR
fi

# check direnv allow status
if command -v direnv &>/dev/null; then
    rc_allowed=$(direnv status 2>/dev/null | grep "RC allowed" | awk '{print $3}')
    if [ "$rc_allowed" != "true" ]; then
        echo "direnv: .envrc not yet allowed. Run: direnv allow ."
    else
        echo "direnv: .envrc is allowed."
    fi
fi

