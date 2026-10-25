import os
import requests
from tqdm import tqdm

raw_path = "../../data/00-raw"
interim_path = "../../data/01-interim"
processed_path = "../../data/02-processed"

def get_raw(file_list, destination_directory):
    """
    Downloads a list of files to a specified destination directory with progress bars.

    Args:
        file_list (list): A list of dictionaries, where each dictionary
                          contains 'url' (the URL of the file) and 'filename'
                          (the desired local filename).
        destination_directory (str): The path to the directory where files
                                     should be saved.
    """
    if not os.path.exists(destination_directory):
        print(f"Error directory {destination_directory} does not exist")
        return

    for file_info in tqdm(file_list, desc="Overall Download Progress"):
        url = file_info['url']
        filename = file_info['filename']
        local_filepath = os.path.join(destination_directory, filename)

        if os.path.exists(local_filepath):
            continue

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            with open(local_filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                          desc=f"Downloading {filename}", leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            pbar.update(len(chunk))
            print(f"Successfully downloaded: {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {filename}: {e}")

if __name__ == "__main__":
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(interim_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)

    datafiles = [
        { "url": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/review-California.json.gz", "filename": "review-California.json.gz"},
        { "url": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/meta-California.json.gz", "filename": "meta-California.json.gz"}
    ]

    get_raw(datafiles, destination_directory="../../data/00-raw/")