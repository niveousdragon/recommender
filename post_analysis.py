import os
import pandas as pd
import json

rpath = os.path.join('C:\\Projects\\temp\\mass test result')

all_counts = []
for df_name in os.listdir(rpath)[:]:
    df = pd.read_excel(os.path.join(rpath, df_name))

    #df = df.groupby('top_cat')
    counts = dict(df['top_cat'].value_counts())
    all_counts.append(counts)


llm_items = pd.read_excel('LLM items corr full v1.1.xlsx')
scs = llm_items['subcategory'].values
tcounts = {}
for sc in scs:
    total = sum([cnt.get(sc, 0) for cnt in all_counts])
    tcounts.update({sc:str(total)})

print(tcounts)
with open('total counts.json', 'w') as json_file:
    json.dump(tcounts, json_file, indent=4)  # 'indent=4' makes the JSON file more readable