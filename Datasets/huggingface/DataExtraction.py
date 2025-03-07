# import json
# import pandas as pd

# # Load the JSON data
# with open('dataset.json') as f:
#     data = json.load(f)

# # Extract rows from the JSON data
# rows = data['rows']

# # Convert rows to a list of dictionaries
# records = [row['row'] for row in rows]

# # Convert the list of dictionaries to a DataFrame
# df = pd.DataFrame(records)

# # Save the DataFrame to a CSV file
# df.to_csv('HuggingFace_NewsDataset.csv', index=False)

# # print("Data has been successfully saved to dataset.csv")
from datasets import load_dataset
import pandas as pd
ds = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")
data = ds['train']
df = data.to_pandas()
df.to_csv('train_Hdata.csv', index=False)
ds["test"].to_pandas().to_csv("test_Hdata.csv", index= False)
print(data)