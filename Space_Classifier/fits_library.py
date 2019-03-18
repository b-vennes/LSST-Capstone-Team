import math
import random
import sys

import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing

import Images

sys.path.append("..")
from Pipeline.Database_Connect import DynamoConnect

IMAGE_SIZE = 16

def parse_images(image_id_list):
    # Open file containing image ids and store them in a list
    sources = []
    with open(image_id_list) as f:
        lines = f.read().splitlines()
    for s_id in lines:
        sources.append(s_id)

    OBJECTS = []
    OBJECTS_LABELS = []

    SPACING = int(IMAGE_SIZE/2)

    scalar = preprocessing.MinMaxScaler()
    scalar.fit([[-100],[1600]])

    # Open fits files for every source and format stars and non-stars
    for fts in sources:
        hdulist = fits.open("Images/" + fts + ".fits")
        # Get data for stars and non-stars
        stars = DynamoConnect.get_stars(fts)
        non_stars = DynamoConnect.get_non_stars(fts)

        fits_data = hdulist[1].data

        print("processing", fts)

        # Each source has x and y WSC - take that and convert to pixel locations
        for star in stars:
            px = int(round(float(star.get('x')) - hdulist[1].header['CRVAL1A']))
            py = int(round(float(star.get('y')) - hdulist[1].header['CRVAL2A']))

            object_array = fits_data[(px-SPACING):(px+SPACING),(py-SPACING):(py+SPACING)]

            if object_array.shape != (IMAGE_SIZE,IMAGE_SIZE):
                continue
            
            for transformation in get_array_transformations(object_array):
                #transformation = scalar.transform(transformation)
                OBJECTS.append(transformation)
                OBJECTS_LABELS.append(1)

        for nstar in non_stars:
            px = int(round(float(nstar.get('x')) - hdulist[1].header['CRVAL1A']))
            py = int(round(float(nstar.get('y')) - hdulist[1].header['CRVAL2A']))

            object_array = fits_data[(px-SPACING):(px+SPACING),(py-SPACING):(py+SPACING)]

            if object_array.shape != (IMAGE_SIZE,IMAGE_SIZE):
                continue

            for transformation in get_array_transformations(object_array):
                #transformation = scalar.transform(transformation)
                OBJECTS.append(transformation)
                OBJECTS_LABELS.append(0)

        hdulist.close()
    
    training_set, validation_set, training_labels, validation_labels = train_test_split(OBJECTS, OBJECTS_LABELS, test_size=0.2, shuffle=True)

    print(np.amax(training_set), np.amax(validation_set))
    print(np.amin(training_set), np.amin(validation_set))

    training_stars = []
    training_nstars = []
    
    for i in range(len(training_labels)):
        if training_labels[i] == 1:
            training_stars.append(training_set[i])
        else:
            training_nstars.append(training_set[i])

    training_stars = np.stack(training_stars, axis=0)
    training_stars = np.expand_dims(training_stars, axis=3)

    training_nstars = np.stack(training_nstars, axis=0)
    training_nstars = np.expand_dims(training_nstars, axis=3)

    validation_set = np.stack(validation_set, axis=0)
    validation_set = np.expand_dims(validation_set, axis=3)

    validation_labels = np.stack(validation_labels, axis=0)
    validation_labels = np.expand_dims(validation_labels, axis=1)

    return training_stars, training_nstars, validation_set, validation_labels

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
