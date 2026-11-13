#!/bin/bash

sudo perf record -a -F 99 -g -- ./build/./filtration-station
sudo perf script > perf.script
stackcollapse-perf.pl perf.script | flamegraph.pl > flamegraph.svg
