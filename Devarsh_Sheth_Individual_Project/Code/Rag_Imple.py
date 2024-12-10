import pandas as pd

# Load the files to inspect their structure
file_russian = "/project/presidential-speeches-rag/Final-Project-Group1/speeches_russian_PM.xlsx"
file_american = "/project/presidential-speeches-rag/Final-Project-Group1/speeches.xlsx"

# Load the Excel files
russian_speeches = pd.ExcelFile(file_russian)
american_speeches = pd.ExcelFile(file_american)

# Check sheet names for both files
russian_sheet_names = russian_speeches.sheet_names
american_sheet_names = american_speeches.sheet_names

# Load data from the first sheet of both files
russian_data = russian_speeches.parse("Sheet1")
american_data = american_speeches.parse("Sheet1")

# Display the first few rows of both datasets
russian_data_head = russian_data.head()
american_data_head = american_data.head()

print(russian_data_head)
print(american_data_head)

