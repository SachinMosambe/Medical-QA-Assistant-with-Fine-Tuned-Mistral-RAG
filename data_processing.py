import pandas as pd
import os

# Predefined instruction
INSTRUCTION = "Answer the question truthfully, you are a medical professional."

def clean_text(text):
    if pd.isna(text):
        return False
    # Filter very short text
    if len(str(text).strip()) < 5:
        return False
    # Filter if text contains only special characters or digits
    if all(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\1234567890" for char in str(text).strip()):
        return False
    # Remove extra spaces and check if text is empty after trimming
    text = str(text).strip()
    if text == "":
        return False
    return True

# Load datasets
df1 = pd.read_csv(r"data\raw_data\medical_meadow_wikidoc.csv")  # Use raw string for path
df2 = pd.read_csv(r"data\raw_data\medquad.csv")  # Use raw string for path

# Normalize columns to have 'input' and 'output'
df1['input'] = df1['input']  # Assuming 'instruct' is the input text
df1['output'] = df1['output']   # Assuming 'output' is the answer

df2['input'] = df2['question']  # Assuming 'question' is the input text
df2['output'] = df2['answer']   # Assuming 'answer' is the output text

# Create the 'instruction' column and set it to the predefined instruction
df1['instruction'] = INSTRUCTION
df2['instruction'] = INSTRUCTION

# Combine datasets
combined_df = pd.concat([df1[['instruction', 'input', 'output']], df2[['instruction', 'input', 'output']]], ignore_index=True)

# Remove duplicates
combined_df = combined_df.drop_duplicates()

# Apply cleaning filters
clean_mask = combined_df.apply(lambda row: clean_text(row['input']) and clean_text(row['output']), axis=1)
cleaned_df = combined_df[clean_mask].reset_index(drop=True)

# Create a 'data' folder if it doesn't exist
output_folder = r"data\process_data"
os.makedirs(output_folder, exist_ok=True)


cleaned_df.to_json(os.path.join(output_folder, "cleaned_combined_dataset.jsonl"), orient='records', lines=True)

print("Data processed and saved in the 'data' folder.")
