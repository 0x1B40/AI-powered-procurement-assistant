import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load sample
data_path = 'archive/PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv'
total_rows = 919734
sample_size = 50000
skip_rows = sorted(np.random.choice(range(1, total_rows), total_rows - sample_size - 1, replace=False))

dtype_dict = {
    'Fiscal Year': 'category',
    'Acquisition Type': 'category',
    'Acquisition Method': 'category',
    'Department Name': 'category',
    'Supplier Name': 'string',
    'CalCard': 'category'
}

print("Loading sample dataset...")
df = pd.read_csv(data_path, dtype=dtype_dict, parse_dates=['Creation Date', 'Purchase Date'],
                 date_format='%m/%d/%Y', skiprows=skip_rows, low_memory=False)

# Clean and convert Total Price column
df['Total Price'] = df['Total Price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

print(f'Successfully loaded sample: {len(df)} rows')
print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB')
print(f'Top acquisition types: {df["Acquisition Type"].value_counts().head(3).to_dict()}')
print(f'Total spending in sample: ${df["Total Price"].sum():,.2f}')
print(f'Date range: {df["Creation Date"].min()} to {df["Creation Date"].max()}')
print(f'Unique departments: {df["Department Name"].nunique()}')
