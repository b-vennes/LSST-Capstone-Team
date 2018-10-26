import script
import os
import pytest
import boto3

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
    """
    Verify that an image id can be posted to the database without error
    """

    image_id = "comet"

    # try to update database
    script.post_database(image_id)

def test_get_random_image():
    """
    Verify that the random url function can be run without error
    """

    # for now just run and hope for no errors
    script.get_random_image_url()