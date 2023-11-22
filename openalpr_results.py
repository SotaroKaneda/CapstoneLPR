import pandas as pd
import csv

preds_df = pd.read_csv("preds.csv")
no_lp_df = pd.read_csv("no_LP.csv")
open_alpr_df = pd.read_csv("../../openalpr_results.csv")

images = open_alpr_df["image"]
preds = open_alpr_df["prediction"]

found = 0
true = 0
false = 0

for image, pred in zip(images, preds):
    look1 = preds_df.loc[preds_df["Image"] == image]
    look2 = no_lp_df.loc[no_lp_df["image"] == image]
    

    if not look1.empty: 
        found = preds_df.loc[preds_df["Image"] == image]["Actual"]
    elif not look2.empty: 
        found = no_lp_df.loc[no_lp_df["image"] == image]["Actual"]

    # print(found, pred)
    if (found == pred).bool():
        true += 1
    else:
        false += 1

print(f"True: {true}\tFalse: {false}")






# look1 = s0_df.loc[s0_df["IMAGE1"] == image_name]
# look2 = s0_df.loc[s0_df["IMAGE2"] == image_name]