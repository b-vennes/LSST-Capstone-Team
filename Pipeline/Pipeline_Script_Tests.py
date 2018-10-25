import script
import os
import pytest
import boto3
from shutil import copyfile

'''
    Description:
    Verify that script.create_image() can create an image
'''
"""
def test_create_image():
    script.create_image()
    emptyFile = False
    for filename in os.listdir('Images'):
        statinfo = os.stat('Images/'+filename)
        if statinfo.st_size == 0:
            emptyFile = True
    assert emptyFile == False and len(os.listdir('Images')) != 0
"""

def test_post_image_to_bucket():
    """
    Verify that script.post_images() successfully sends an image to s3 bucket
    """

    image = "comet.jpg"

    # post image to s3, error thrown if unsuccessful
    link = script.post_image(image)

    # check that link has been formatted correctly
    assert(link == "https://s3-us-west-2.amazonaws.com/lsst-images/comet.jpg")

    # delete image from s3, error will be thrown if unsuccessful
    s3 = boto3.resource('s3')
    s3.Object('lsst-images', image).delete()

def test_post_image_to_dynamo():

    image_id = "comet"

    # try to update database
    script.post_database(image_id)