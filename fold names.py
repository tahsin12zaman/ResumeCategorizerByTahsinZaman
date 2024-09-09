import os


def list_folders(directory):
    # Get a list of all entries in the directory
    entries = os.listdir(directory)

    # Filter out entries that are folders
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

    return folders


# Specify the directory you want to check
directory_path = '/home/user/Desktop/Projects/ResumeCategorizerByTahsinZaman/dataset/data'

# Get the list of folder names
folders = list_folders(directory_path)

# Print the folder names and count
print("Folders in the directory:")
for folder in folders:
    print(folder)

print(f"\nTotal number of folders: {len(folders)}")
