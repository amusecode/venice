#!/bin/csh

if (! -d data) then
    mkdir data
endif
if (! -d figures) then
    mkdir figures
endif

if ( "$1" != "" ) then
    mpirun python paper_gravostellar.py --mode constant --filepath "$1"
    mpirun python paper_gravostellar.py --mode reversed --filepath "$1"
    mpirun python paper_gravostellar.py --mode adaptive --filepath "$1"
else
    echo "Simulation snapshots will be written to repository directory"
    echo "These may be large/many!"
    mpirun python paper_gravostellar.py --mode constant
    mpirun python paper_gravostellar.py --mode reversed
    mpirun python paper_gravostellar.py --mode adaptive
endif

mpirun python paper_bridge.py --eccentricity 0.0
mpirun python paper_bridge.py --eccentricity 0.5

if ( "$1" != "" ) then
    if (! -d $1/combined_s0_p0) then
        mkdir $1/combined_s0_p0
    endif
    if (! -d $1/combined_s1_p0) then
        mkdir $1/combined_s1_p0
    endif
    if (! -d $1/combined_s0_p1) then
        mkdir $1/combined_s0_p1
    endif
    if (! -d $1/combined_s1_p1) then
        mkdir $1/combined_s1_p1
    endif
    mpirun python paper_combined.py -s 0 -p 0 --filepath "$1"
    mpirun python paper_combined.py -s 1 -p 0 --filepath "$1"
    mpirun python paper_combined.py -s 0 -p 1 --filepath "$1"
    mpirun python paper_combined.py -s 1 -p 1 --filepath "$1"
else
    if (! -d data/combined_s0_p0) then
        mkdir data/combined_s0_p0
    endif
    if (! -d data/combined_s1_p0) then
        mkdir data/combined_s1_p0
    endif
    if (! -d data/combined_s0_p1) then
        mkdir data/combined_s0_p1
    endif
    if (! -d data/combined_s1_p1) then
        mkdir data/combined_s1_p1
    endif
    mpirun python paper_combined.py -s 0 -p 0
    mpirun python paper_combined.py -s 1 -p 0
    mpirun python paper_combined.py -s 0 -p 1
    mpirun python paper_combined.py -s 1 -p 1
endif
