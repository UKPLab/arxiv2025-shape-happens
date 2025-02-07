import os
import pandas as pd
import argparse

def read_csv_robust(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf8")
    except Exception as e:
        try:
            return pd.read_csv(file_path, encoding='cp1252')
        except Exception as e:
            raise Exception(f"Failed to read {file_path} due to {e}")
        

def scan_directory(directory):
    dataset = pd.DataFrame()
    all_columns = set()
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                temp_df = read_csv_robust(file_path)
                
                if temp_df is not None:
                    all_columns.update(temp_df.columns)
                    dataset = pd.concat([dataset, temp_df], ignore_index=True, sort=False)
    
    # Ensure all missing columns are filled with None
    for col in all_columns:
        if col not in dataset.columns:
            dataset[col] = None
    
    return dataset

def filter_questions(df):
    # Keep only questions without 'Premise' and 'Hypothesis' fields
    return df[df['Premise'].isnull() & df['Hypothesis'].isnull()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan directory for CSV files and merge them into a dataset.")
    parser.add_argument("directory", help="Directory to scan for CSV files.", type=str, default=".") 
    args = parser.parse_args()
    
    dataset = scan_directory(args.directory)
    dataset = filter_questions(dataset)
    print(dataset.head())  # Print first few rows to verify
    output_file = os.path.join(args.directory, "merged_dataset.csv")
    dataset.to_csv(output_file, index=False)  # Save to CSV if needed
