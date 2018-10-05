import script
import pytest
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
