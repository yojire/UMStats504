"""
Create a map called dcat that maps drug categories to lists of drug
names belonging to the category.
"""

import pandas as pd

dc = pd.read_excel("PartD_Prescriber_PUF_NPI_15_Drug_Category_Lists.xlsx", sheet_name=None)

mp = {"Antibiotic Drug Names": "antibiotic",
      "Antipsychotic Drug Names": "antipsychotic",
      "Opioid Drug Names": "opioid",
      "High-Risk Medication Drug Names": "high_risk"}

dcat = {m: set([]) for _,m in mp.items()}

for v,m in mp.items():
    sheet = dc[v].iloc[3:, :]
    for i in range(sheet.shape[0]):
        x = sheet.iloc[i, 1]
        if pd.isnull(x):
            break
        dcat[m].add(x)

for k,v in dcat.items():
    dcat[k] = list(v)
    dcat[k].sort()
