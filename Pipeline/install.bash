#!/bin/bash
mkdir -p lsst_stack
cd lsst_stack
curl -OL https://raw.githubusercontent.com/lsst/lsst/16.0/scripts/newinstall.sh
bash newinstall.sh -ct
source loadLSST.bash
eups distrib install -t v16_0 lsst_distrib
curl -sSL https://raw.githubusercontent.com/lsst/shebangtron/master/shebangtron | python
setup lsst_distrib
cd ..
cp -a /bash_scripts/. /lsst_stack/
