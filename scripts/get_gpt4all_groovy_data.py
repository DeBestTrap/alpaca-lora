# %%
import os, sys
from datasets import load_dataset


def get_groovy_data():
  '''
  Save v1.3 Groovy data from gpt4all in a the data directory 

  Returns:
    True on success
    False on failure
  '''

  # Find root directory
  # There might be a better way to do this
  if os.path.exists(".gitignore"):
    path = "."
  elif os.path.exists("../.gitignore"):
    path = ".."
  else:
    print("Unable to find root directory", file=sys.stderr)
    return False 

  # Create data directory if it does not exist
  data_dir = os.path.join(path, "data/")
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

  # Fetch and save data
  DATA_PATH = "nomic-ai/gpt4all-j-prompt-generations"
  data = load_dataset(DATA_PATH, revision="v1.3-groovy")
  data = data.rename_column("prompt", "instruction")
  data = data.rename_column("response", "output")
  data = data.remove_columns("source")
  data = data["train"].add_column("input", ["" for i in data["train"]])
  data.to_json(os.path.join(data_dir, "v1.3-groovy.json"))

  print("Groovy v1.3 data saved successfully in alpaca format.")
  return True

if __name__ == "__main__":
  get_groovy_data()