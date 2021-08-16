#!/bin/bash

## Test python environment is setup correctly
if [[ $1 = "test_environment" ]]; then
	echo ">>> Testing Python Environment"
	/usr/local/bin/test_environment.py
fi

## Install Python Dependencies
if [[ $1 = "requirements" ]]; then
	echo ">>> Installing Required Modules .."
	cd /usr/local/bin/
	pip3 install -U pip setuptools wheel
	echo ">>> Done!"
fi
