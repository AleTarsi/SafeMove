#!/bin/bash

# Check if SAFE_MOVE_PATH is already set
if [ "${SAFE_MOVE_PATH}" != "$(realpath $(pwd))" ]; then
    # If not, append the line to .bashrc
    echo "export SAFE_MOVE_PATH=$(realpath "$(pwd)")" >> safemove.sh
	source safemove.sh
	echo "source $SAFE_MOVE_PATH/safemove.sh" >> ~/.bashrc
	echo "Created $SAFE_MOVE_PATH env. variable"
else 
	echo "SAFE_MOVE_PATH correctly set :)"
fi