import requests
import script
import os
from shutil import copyfile

'''
    Test Description:
    Verify that script.create_image() can create an image
'''
def test_create_image():
    script.create_image()
    emptyFile = False
    for filename in os.listdir('Images'):
        statinfo = os.stat('Images/'+filename)
        if statinfo.st_size == 0:
            emptyFile = True
    assert emptyFile == False and len(os.listdir('Images')) != 0

'''
    Test Description:
    Verify that script.post_images() successfully sends an image to the API
'''
def test_post_images():

    srcFile = "spooky.jpg"
    newFile = "toospooky.jpg"

    # make a copy of the source image
    copyfile(srcFile, newFile)

    # post image to API and check for success
    postSuccess = script.post_images(newFile)
    assert postSuccess

    # check if file was deleted
    assert (os.path.isfile(newFile) != True)