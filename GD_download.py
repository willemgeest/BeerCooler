#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
from pathlib import Path


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def get_beerclass_model_Drive(modelname):
    f_checkpoint = Path(modelname)
    if not f_checkpoint.exists():
        download_file_from_google_drive('1A9qhi2EpkfC9pAK_l9rWNWPMjwM4ZPVr', f_checkpoint)
