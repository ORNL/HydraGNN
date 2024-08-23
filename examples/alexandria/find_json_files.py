import os
import shutil
import subprocess

import requests
from bs4 import BeautifulSoup


def find_json_files(url):
    response = requests.get(url)
    response.raise_for_status()  # Raises an error on a bad status

    soup = BeautifulSoup(response.text, "html.parser")
    json_files = [
        link.get("href")
        for link in soup.find_all("a")
        if link.get("href").endswith(".bz2")
    ]

    return json_files


indices = ["pascal", "pbe", "pbe_1d", "pbe_2d", "pbesol", "scan"]

url_root = "https://alexandria.icams.rub.de/data"  # Replace with the actual URL

dirpath = "dataset/compressed_data"

if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

os.makedirs(dirpath)

for index in indices:

    json_files = find_json_files(os.path.join(url_root, index))
    new_dir = os.path.join(dirpath, index)
    os.makedirs(new_dir)

    for json_file in json_files:
        subprocess.run(
            [f"wget -nv {url_root+'/'+index+'/'+json_file} -P {new_dir}"],
            shell=True,
            check=True,
            executable="/bin/bash",
        )
        print(json_file)
