#!/bin/bash

# 1. First check to see if the correct version of Python is installed on the local machine
#echo "1. Checking Python version..."
#REQ_PYTHON_V="380"

#ACTUAL_PYTHON_V=$(python -c 'import sys; version=sys.version_info[:3]; print("{0}{1}{2}".format(*version))')
#ACTUAL_PYTHON3_V=$(python3 -c 'import sys; version=sys.version_info[:3]; print("{0}{1}{2}".format(*version))')

#if [[ $ACTUAL_PYTHON_V > $REQ_PYTHON_V ]] || [[ $ACTUAL_PYTHON_V == $REQ_PYTHON_V ]];  then
#    PYTHON="python"
#elif [[ $ACTUAL_PYTHON3_V > $REQ_PYTHON_V ]] || [[ $ACTUAL_PYTHON3_V == $REQ_PYTHON_V ]]; then
#    PYTHON="python3"
#else
#    echo -e "\tPython 3.8 is not installed on this machine. Please install Python 3.8 before continuing."
 #   exit 1
#fi

echo -e "\t--Python 3.98is installed"

# 2. What OS are we running on?
PLATFORM=$($PYTHON -c 'import platform; print(platform.system())')

echo -e "2. Checking OS Platform..."
echo -e "\t--OS=Platform=$PLATFORM"

# 3. Create Virtual environment
echo -e "3. Creating new virtual environment..."

# Remove the env directory if it exists
if [[ -d jj_dt ]]; then
    echo -e "\t--Virtual Environment already exists. Deleting old one now."
    rm -rf jj_dt
fi

$PYTHON -m venv jj_dt
if [[ ! -d jj_dt ]]; then
    echo -e "\t--Could not create virtual environment...Please make sure venv is installed"
    exit 1
fi

# 4. Install Requirements

echo -e "4. Installing Requirements..."
if [[ ! -e "jj_dt_project_requirements.txt" ]]; then
    echo -e "\t--Need jj_dt_project_requirements to install packages."
    exit 1
fi

source jj_dt/bin/activate
pip install -r jj_dt_project_requirements.txt