import script
from shutil import copyfile

def test_create_image():
    '''
    Verify that script.create_image() can create an image
    '''

    script.create_image()
    emptyFile = False
    for filename in os.listdir('Images'):
        statinfo = os.stat('Images/'+filename)
        if statinfo.st_size == 0:
            emptyFile = True
    assert emptyFile == False and len(os.listdir('Images')) != 0