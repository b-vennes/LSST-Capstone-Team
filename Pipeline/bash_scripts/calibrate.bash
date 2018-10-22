#!/bin/bash
eups list lsst_distrib
processCcd.py DATA --rerun processCcdOutputs --id --show data
processCcd.py DATA --rerun processCcdOutputs --id filter=HSC-I --show data
processCcd.py DATA --rerun processCcdOutputs --id
