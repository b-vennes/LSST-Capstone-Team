import DynamoConnect
import os
import pytest
import boto3
import urllib.request as urlreq

def test_post_image():
    """
    Verify that script.post_images() successfully sends an image to s3 bucket.
    """

    image = "comet.jpg"

    # post image to s3, error thrown if unsuccessful
    link = DynamoConnect.post_image(image)

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
    source = "testing"

    # try to update database
    DynamoConnect.post_database(image_id, source)

def test_random_image():
    """
    Verify that the random url function can be run without error.
    """

    # for now just run and hope for no errors
    DynamoConnect.random_image()

def test_upload_image():
    """
    Verify that the upload pipeline works by uploading an image and downlaoding it.
    """

    image_name = 'test.jpg'
    source_name = 'testing'

    # create an empty image file and upload to data center
    open(image_name, 'w')
    link = DynamoConnect.upload_image(image_name, source_name)

    os.remove(image_name)

    # pull down the image from the link and check that it downloaded successfully
    image_destination = "pulled_image.jpg"
    urlreq.urlretrieve(link, image_destination)
    assert os.path.isfile(image_destination)
    
    os.remove(image_destination)

    




