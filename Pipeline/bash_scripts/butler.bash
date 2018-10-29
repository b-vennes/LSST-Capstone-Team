#!/bin/bash
cd ../../lsst_stack
git clone https://github.com/lsst/ci_hsc
setup -j -r ci_hsc
echo $CI_HSC_DIR
mkdir DATA
echo "lsst.obs.hsc.HscMapper" > DATA/_mapper
ingestImages.py DATA $CI_HSC_DIR/raw/*.fits --mode=link
ln -s $CI_HSC_DIR/CALIB/ DATA/CALIB
mkdir -p DATA/ref_cats
ln -s $CI_HSC_DIR/ps1_pv3_3pi_20170110 DATA/ref_cats/ps1_pv3_3pi_20170110
