import os
import zipfile
import requests

isf_data_id = '1L9ydPdjF9XzMfK3rXJ93_1fW1OS8YiGc'
gtsrb_data_id = '1NVQLLAqIup4p_Rcn_JJzB905Qyk4ICYI'


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm' : token }
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


def get_necessary_data(dataset_name, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(data_dir + os.sep + dataset_name):
        print('downloading..')
        if dataset_name == 'isf':
            download_file_from_google_drive('1Xvw7w3XKNLPWwfCZMKordtS7c-sIc_cs', data_dir + os.sep + 'data.zip')
        elif dataset_name == 'gtsrb':
            download_file_from_google_drive('1pD3PhvGhd8vXRazfzCjXe7dkNvXiUVtc', data_dir + os.sep + 'data.zip')
        print('unzipping..')
        zip_ref = zipfile.ZipFile(data_dir + os.sep + 'data.zip', 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()
        print('deleting zip..')
        os.remove(data_dir + os.sep + 'data.zip')
    return data_dir + os.sep + dataset_name + os.sep + 'train', data_dir + os.sep + dataset_name + os.sep + 'test'


if __name__ == "__main__":
    file_id = '1L9ydPdjF9XzMfK3rXJ93_1fW1OS8YiGc'
    destination = './test.zip'
    download_file_from_google_drive(file_id, destination)