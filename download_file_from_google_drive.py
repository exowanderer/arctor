import requests


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

if __name__ == "__main__":
    import os
    import errno

    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    joblib_file_id = '1x8tu49oIj1f9Sd007s37xK4O4NRTG14b'
    destination = 'wasp43_savedict_206ppm_100x100_finescale.joblib.save'
    download_file_from_google_drive(joblib_file_id, destination)

    tar_file_id = '1TscP7ES99XAWeHYT21JJBGbiKn1C8iFs'
    destination = 'WASP_HST_UVIS_FLT_Fits_Files.tar.gz'
    download_file_from_google_drive(tar_file_id, destination)

    spitzercal_file_id = '1e4iZ0vLPq9w-TeZJcCIConiXKqsjC89X'
    destination = 'pmap_ch2_0p1s_x4_rmulti_s3_7.csv'
    download_file_from_google_drive(tar_file_id, destination)

    # SpitzerCal for the GA-CNNs
    spitzercal_file_id = '1e4iZ0vLPq9w-TeZJcCIConiXKqsjC89X'
    HOME = os.environ['HOME']
    dest_dir = f'{HOME}/.vaelstmpredictor/data/'

    if not os.path.exists(dest_dir):
        mkdir_p(dest_dir)

    destination = os.path.join(dest_dir, 'pmap_ch2_0p1s_x4_rmulti_s3_7.csv')
    download_file_from_google_drive(spitzercal_file_id, destination)
