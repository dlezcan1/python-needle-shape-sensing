#!/bin/bash

python3 /home/dlezcan1/dev/git/python-needle-shape-sensing/scripts/run_stochastic_model.py \
    --needle-json /home/dlezcan1/dev/git/python-needle-shape-sensing/data/stochastic_model_test.json \
    --odir /tmp/stochastic-shape-results \
    --kc 0.0025508 \
    --insertion-depth 90 \
    --ds 0.5 \
    --dw 0.002 \
    --w-bounds -0.05 0.05 \
    --std-curvature 0.0005 \
    $@