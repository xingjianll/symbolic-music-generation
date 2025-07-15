from huggingface_hub import create_repo, upload_folder

# # Create repo if it doesn't exist
# create_repo("xingjianll/midi-gpt2", private=True)

# Upload the folder
upload_folder(
    folder_path="../midi-gpt2",
    path_in_repo="..",
    repo_id="xingjianll/midi-gpt2",
    repo_type="model"
)


if __name__ == "__main__":
    print("Done")