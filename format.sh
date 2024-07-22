#!/bin/bash

python3 -m venv $HOME/.venv
source $HOME/.venv/bin/activate

if ! pip list | grep black; then
    pip install black
fi

black --force-exclude hex_llm/models/kernels ./

deactivate