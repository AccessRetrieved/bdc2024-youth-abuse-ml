import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Data.csv")

selected_vars = [
    'SS_010',
    'ALC_010',
    'CAN_010',
    'ELC_026a',
    'GH_010',
    'GH_020',
    'BEH_010',
    'GRADE',
    'DVURBAN',
    'DVCIGWK',
    'PH_010'
]

heatmap_data = data[selected_vars]

heatmap_data = heatmap_data.fillna(heatmap_data.mode().iloc[0])

for col in ['SS_010', 'ALC_010', 'CAN_010', 'ELC_026a', 'BEH_010']:
    heatmap_data[col] = heatmap_data[col].replace({'Yes': 1, 'No': 0})

correlation_matrix = heatmap_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()