import requests
import script
import os

def test_create_image():
    script.create_image()
    emptyFile = False
    for filename in os.listdir('Images'):
        statinfo = os.stat('Images/'+filename)
        if statinfo.st_size == 0:
            emptyFile = True
    assert emptyFile == False and len(os.listdir('Images')) != 0

def test_push_images():
    script.push_images()
assert False

'''
    Test Description:
    Attempt to reach the API and GET a 200 (OK) response.
    Tests that the API can be connected to and is alive.
'''
def test_api_get_response():
    data = requests.get("http://localhost:7071/api/lsst-label-api")
    assert(data.status_code == 200)

'''
    Test Description:
    Attempt to send nothing to the API, the API should return an error.
'''
def test_api_post_without_image():
    data = requests.post("http://localhost:7071/api/lsst-label-api")
    assert(data.status_code != 200)

'''
    Test Description:
    Send an actual image and expect a 200 response.
'''
def test_send_image_to_api():
    with open("spooky.jpg", "rb") as imageFile:
        f = imageFile.read()
        b = bytearray(f)

    data = requests.post("http://localhost:7071/api/lsst-label-api", data = b)
    assert(data.status_code == 200)