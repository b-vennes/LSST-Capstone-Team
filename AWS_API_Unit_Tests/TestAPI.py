import json
import requests

"""
Unit tests for API
"""

def test_upload_image():
    """
    Verify that the post_image API is alive.
    """

    upload_image_api_url = ""

    api_response = requests.post(upload_image_api_url, {}).json()

    assert api_response["statusCode"] == 200


def test_aws_api_random_image():
    """
    Verify that the random_image API is alive
    """

    random_image_api_url = "https://bypff11t87.execute-api.us-west-2.amazonaws.com/default/random_image"

    api_response = requests.get(random_image_api_url).json()

    assert api_response["statusCode"] == 200

def test_aws_api_update_label():
    """
    Verify that the post_label API is alive
    """

    post_label_api_url = ""

    api_response = requests.post(post_label_api_url, data = {}).json()

    assert api_response["statusCode"] == 200
