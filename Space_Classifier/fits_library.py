import math
import random
import sys

import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib import pyplot as plt

import Images

sys.path.append("..")
from Pipeline.Database_Connect import DynamoConnect

IMAGE_SIZE = 32

def parse_images(image_id_list):
    # Open file containing image ids and store them in a list
    sources = []
    with open(image_id_list) as f:
        lines = f.read().splitlines()
    for s_id in lines:
        sources.append(s_id)

    ALL_STARS = []
    ALL_NON_STARS = []

    SPACING = int(IMAGE_SIZE/2)

    # Open fits files for every source and format stars and non-stars
    for fts in sources:
        hdulist = fits.open("Images/" + fts + ".fits")
        # Get data for stars and non-stars
        stars = DynamoConnect.get_stars(fts+".fits")
        non_stars = DynamoConnect.get_non_stars(fts+".fits")

        fits_data = hdulist[3].data

        print("processing", fts)

        # Each source has x and y WSC - take that and convert to pixel locations
        for star in stars:
            px = int(round(float(star.get('x')) - hdulist[3].header['CRVAL1A']))
            py = int(round(float(star.get('y')) - hdulist[3].header['CRVAL2A']))

            object_array = fits_data[(px-SPACING):(px+SPACING),(py-SPACING):(py+SPACING)]

            if object_array.shape != (IMAGE_SIZE,IMAGE_SIZE):
                continue

            ALL_STARS.append(object_array)

        for nstar in non_stars:
            px = int(round(float(nstar.get('x')) - hdulist[3].header['CRVAL1A']))
            py = int(round(float(nstar.get('y')) - hdulist[3].header['CRVAL2A']))

            object_array = fits_data[(px-SPACING):(px+SPACING),(py-SPACING):(py+SPACING)]

            if object_array.shape != (IMAGE_SIZE,IMAGE_SIZE):
                continue

            ALL_NON_STARS.append(object_array)

        hdulist.close()
            
    # Seperate 80% of true-positives and true-negatives for training
    # Seperate other 20% for validation
    num_star_training = int(len(ALL_STARS) * 0.8)
    num_star_validation = len(ALL_STARS) - num_star_training
    num_nstar_training = int(len(ALL_NON_STARS) * 0.8)
    num_nstar_validation = len(ALL_NON_STARS) - num_nstar_training
    
    # Create training and validation lists
    training_set = []
    training_labels = []
    validation_set = []
    validation_labels = []
    for i in range(0,num_star_training):

        for transformation in get_array_transformations(ALL_STARS[i]):
            training_set.append(transformation)

        for i in range(8):
            training_labels.append(1)

    for i in range(0,num_nstar_training):
            
        for transformation in get_array_transformations(ALL_NON_STARS[i]):
            training_set.append(transformation)

        for i in range(8):
            training_labels.append(0)

    for i in range(num_star_training, num_star_training + num_star_validation):
        
        for transformation in get_array_transformations(ALL_STARS[i]):
            validation_set.append(transformation)

        for i in range(8):
            validation_labels.append(1)

    for i in range(num_nstar_training, num_nstar_training + num_nstar_validation):
        
        for transformation in get_array_transformations(ALL_NON_STARS[i]):
            validation_set.append(transformation)

        for i in range(8):
            validation_labels.append(0)

    training_set = np.stack(training_set, axis=0)
    training_set = np.expand_dims(training_set, axis=3)

    validation_set = np.stack(validation_set, axis=0)
    validation_set = np.expand_dims(validation_set, axis=3)

    training_labels = np.stack(training_labels, axis=0)
    training_labels = np.expand_dims(training_labels, axis=1)

    validation_labels = np.stack(validation_labels, axis=0)
    validation_labels = np.expand_dims(validation_labels, axis=1)

    return training_set, training_labels, validation_set, validation_labels

def get_array_transformations(arr):
    transformation = []

    orig_rot90 = np.rot90(arr)
    orig_rot180 = np.rot90(orig_rot90)
    orig_rot270 = np.rot90(orig_rot180)
    flip = np.flip(arr)
    flip_rot90 = np.rot90(flip)
    flip_rot180 = np.rot90(flip_rot90)
    flip_rot270 = np.rot90(flip_rot180)

    transformation.append(arr)
    transformation.append(orig_rot90)
    transformation.append(orig_rot180)
    transformation.append(orig_rot270)
    transformation.append(flip)
    transformation.append(flip_rot90)
    transformation.append(flip_rot180)
    transformation.append(flip_rot270)

    return transformation
