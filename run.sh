#!/bin/bash

python make-data.py --dim 4096

make

nvprof ./bin/cbert 0 # run without cub
nvprof ./bin/cbert 1 # run with cub