import zipfile
import os

# Path to the zip file
zip_file_path = r"C:\Users\haito\Desktop\technion\semester 8\lab\final_project\data\archive (1).zip"

# Destination directory where you want to extract the files
extract_dir = r"C:\Users\haito\Desktop\technion\semester 8\lab\final_project\data"

# Make sure the directory exists
os.makedirs(extract_dir, exist_ok=True)

# Unzipping the folder
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f'Files unzipped to: {extract_dir}')