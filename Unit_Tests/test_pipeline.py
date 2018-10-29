import requests

# attempt to reach the API and GET a 200 (OK) response
def test_api_get_connection():
    data = requests.get("http://localhost:7071/api/lsst-label-api")
    assert(data.status_code == 200)

# attempt to reach the API's POST method and get a 200 (OK) response
def test_api_post_connection():
    data = requests.post("http://localhost:7071/api/lsst-label-api")
    assert(data.status_code == 200)