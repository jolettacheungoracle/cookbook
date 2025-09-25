#!/usr/bin/env bash

echo -----------------------
echo ENV
if [ -z "$1" ]
  then
    echo "using default local.env"
    env_file="./local.env"
  else
    echo "using " $1
    env_file="$1"
fi

#env | sort
set -a; source $env_file; set +a
echo -----------------------
echo PYTHON
echo activating python virtual environment at venv/bin/activate
source ../.venv/bin/activate
echo -----------------------
