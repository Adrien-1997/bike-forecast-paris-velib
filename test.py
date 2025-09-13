import pandas as pd
df=pd.read_parquet('docs/exports/velib.parquet')
print(df['tbin_utc'].min(), '->', df['tbin_utc'].max(), 'rows=',len(df))