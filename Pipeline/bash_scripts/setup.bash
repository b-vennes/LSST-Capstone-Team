#!/bin/bash
cd ../../lsst_stack
source loadLSST.bash
setup lsst_distrib
eups list -s
