import database_connect_module
import os
import pytest
import boto3
import urllib

def test_post_image():
    """
    Verify that script.post_images() successfully sends an image to s3 bucket.
    """

    image = "comet.jpg"

    # post image to s3, error thrown if unsuccessful
    link = database_connect_module.post_image(image)

    # check that link has been formatted correctly
    assert link == "https://s3-us-west-2.amazonaws.com/lsst-images/comet.jpg"

    # delete image from s3, error will be thrown if unsuccessful
    s3 = boto3.resource('s3')
    s3.Object('lsst-images', image).delete()

def test_post_database():
    """
    Verify that an image id can be posted to the database without error.
    """

    image_id = "comet"

    # try to update database
    database_connect_module.post_database(image_id)

def test_random_image():
    """
    Verify that the random url function can be run without error.
    """

    # for now just run and hope for no errors
    database_connect_module.random_image()

def test_upload_image():
    """
    Verify that the upload pipeline works by uploading an image and downlaoding it.
    """

    image_source = 'test.jpg'

    # create an empty image file and upload to data center
    open(image_source, 'w')
    link = database_connect_module.upload_image(source_image_name)

    os.remove(image_source)

    # pull down the image from the link and check that it downloaded successfully
    image_destination = "pulled_image.jpg"
    urllib.URLopener().retrieve(link, image_destination)
    assert os.path.isfile(image_destination)
    
    os.remove(image_destination)

    




