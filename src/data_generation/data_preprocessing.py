import kagglehub
import os
import pandas as pd

def clean_dataset():

    # To get the actual raw file name
    dataset_path = kagglehub.dataset_download("ahmedkallam/ikea-sa-furniture-web-scraping")
    #print("Path to dataset files:", dataset_path) = IKEA_SA_Furniture_Web_Scrapings_sss.csv

    raw_data_path = os.path.join(dataset_path, "IKEA_SA_Furniture_Web_Scrapings_sss.csv")

    df = pd.read_csv(raw_data_path)
    df.columns = df.columns.str.lower() # lowercase

    # keep only defined columns from the data table
    df_clean = df[['item_id', 'name', 'category', 'short_description', 'designer', 'width', 'height', 'depth']] 

    df_clean['short_description'] = df_clean['short_description'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    # all columns were filled with designer name and had descriptions
    #df_clean['designer'] = df_clean['designer'].fillna('Unknown')
    #df_clean['short_description'] = df_clean['short_description'].replace('nan', 'No description')

    # drop exact duplicates based on item ID
    df_clean = df_clean.drop_duplicates(subset=['item_id'])

    df_clean.to_csv('data/processed_ikea_data.csv', index=False)

    print(f"Data saved to data/processed_ikea_data.csv")

if __name__ == "__main__":
    clean_dataset()
