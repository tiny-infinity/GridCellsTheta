#!/bin/bash

for i in {1..25}; do
    echo "Generating simulation setup for sweep $i"
    python s_sim_setup.py specs/phpc/stell_inhib_sweeps_0/PHPC-INHIB-SWEEP-0-$i.py -o

    echo "Setup generation for sweep $i completed"

done