# %%
import os, sys
from datasets import load_dataset

def get_unified_chip2_data():
  '''
  Save unified chip 2 from LAION data in a the data directory 

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
  DATA_PATH = "nomic-ai/gpt4all_prompt_generations"
  data = load_dataset(DATA_PATH)
  data = data.filter(lambda x: x["source"] == "unified_chip2")
  data = data.rename_column("prompt", "instruction")
  data = data.rename_column("response", "output")
  data = data.remove_columns("source")
  data = data["train"].add_column("input", ["" for i in data["train"]])
  data.to_json(os.path.join(data_dir, "unified_chip2.json"))

  print("Unified chip 2 data saved successfully in alpaca format.")
  return True

if __name__ == "__main__":
  get_unified_chip2_data()