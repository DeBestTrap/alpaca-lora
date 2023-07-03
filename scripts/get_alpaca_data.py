# %%
import requests, json, os, sys

def get_alpaca_data():
  '''
  Save alpaca data in a the data directory 

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

  def alpaca_helper(data_dir, data_path):
    if not os.path.exists(os.path.join(data_dir, data_path)):
      link = f"https://raw.githubusercontent.com/tloen/alpaca-lora/e04897baaec39280fac97f1ad2bf33059b0df643/{data_path}"
      req = requests.get(link)
      with open(os.path.join(data_dir, data_path), "w") as f:
        json.dump(req.text, f)
        print(f"{data_path} saved successfully.")

  # Fetch and save data
  data_path = "alpaca_data.json"
  alpaca_helper(data_dir, data_path)
  data_path = "alpaca_data_cleaned.json"
  alpaca_helper(data_dir, data_path)
  return True

if __name__ == "__main__":
  get_alpaca_data()
# %%