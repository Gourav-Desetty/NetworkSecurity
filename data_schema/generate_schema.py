import yaml
import pandas as pd

df = pd.read_csv(r"Network_Data\phisingData.csv")

schema = {
    "columns" : [
        {col: str(df[col].dtype)} for col in df.columns
    ],
    "numerical_columns" : list([col for col in df.columns if df[col].dtype != 'O'])
}

with open("data_schema\schema.yaml", 'w') as f:
    yaml.dump(schema, f)