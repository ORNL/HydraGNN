import zipfile
import os

# Define the output directory and zip file path
output_dir = "dataset/JARVIS-DFT"
zip_file_path = os.path.join(output_dir, "6815699.zip")

# Function to unzip a file and maintain the directory structure
def unzip_file(zip_file, extract_to):
    # Check if the zip file exists
    if os.path.exists(zip_file):
        print(f"Extracting {zip_file} to {extract_to} ...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_to)  # Extract the current zip file
        print(f"Extraction complete for {zip_file}.")
    else:
        print(f"Zip file {zip_file} not found!")


# Function to recursively unzip nested zip files while maintaining the original structure
def unzip_nested_zips(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                file_path = os.path.join(root, file)
                extract_dir = (
                    root  # Extract the zip file in the same directory it resides
                )
                print(f"Found nested zip: {file_path}, extracting to {extract_dir}...")
                unzip_file(
                    file_path, extract_dir
                )  # Unzip the nested zip in the same folder


# Call the function to unzip the main zip file
unzip_file(zip_file_path, output_dir)

# Call the function to recursively unzip nested zip files
unzip_nested_zips(output_dir)
