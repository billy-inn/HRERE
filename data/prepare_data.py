from gensim.scripts.glove2word2vec import glove2word2vec
import os
import requests
import tarfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

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
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

print("Downloading NYT corpus...")
download_file_from_google_drive("1A5V1qXZJvPj0uQYVxO0Hl0Dy-FdbPQPq", "NYT.tar.gz")
tar = tarfile.open("NYT.tar.gz", "r:gz")
tar.extractall()
tar.close()
os.remove("NYT.tar.gz")
os.system("mv data/* ./")
os.system("rm -r data/")

print("Downloading fb3m dataset...")
download_file_from_google_drive("1XYdeovP9XuRh_j83R7sSow6mwQhoUdwQ", "fb3m.tar.gz")
tar = tarfile.open("fb3m.tar.gz", "r:gz")
tar.extractall()
tar.close()
os.remove("fb3m.tar.gz")
os.system("mv fb3m/raw.txt triples.csv")
os.system("rm -r fb3m/")

print("Downloading GloVe embeddings...")
os.system("wget http://nlp.stanford.edu/data/glove.840B.300d.zip")
os.system("unzip glove.840B.300d.zip")
os.system("rm glove.840B.300d.zip")

glove_file = "glove.840B.300d.txt"
word2vec_file = "glove.840B.300d.w2v.txt"
_ = glove2word2vec(glove_file, word2vec_file)

os.system("mv %s %s" % (word2vec_file, glove_file))
