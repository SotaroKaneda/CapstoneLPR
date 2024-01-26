import pandas as pd
import csv


# https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
# https://www.geeksforgeeks.org/writing-csv-files-in-python/#
# https://towardsdatascience.com/how-to-select-rows-from-pandas-dataframe-based-on-column-values-d3f5da421e93
# https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html



### save sheets to csv
##################################################################################################
# dataframe = pd.DataFrame(pd.read_excel("../../ALPRPlateExportDaytime.xlsx", sheet_name=0))
# dataframe.to_csv("../../sheet0.csv",  header=True)

# dataframe = pd.DataFrame(pd.read_excel("../../ALPRPlateExportDaytime.xlsx", sheet_name=1))
# dataframe.to_csv("../../sheet_1.csv",  header=True)
##################################################################################################

s0_df = pd.read_csv("../../sheet_0.csv", usecols=["IMAGE1", "IMAGE2", "PLATE_READ"])
s1_df = pd.read_csv("../../sheet_1.csv", usecols=["IMAGE1", "IMAGE2", "PLATE_READ"])


fields = ["PLATE_READ", "IMAGE1", "IMAGE2"]


with open("../../s0_rel.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    
    # Write Sheet 0 data
    for i in range(len(s0_df)):
        image_1 = s0_df.loc[i]["IMAGE1"].split("/")[-1] if type(s0_df.loc[i]["IMAGE1"]) != float else 0
        image_2 = s0_df.loc[i]["IMAGE2"].split("/")[-1] if type(s0_df.loc[i]["IMAGE2"]) != float else 0
        plate_num = s0_df.loc[i]["PLATE_READ"]
        row = [plate_num, image_1, image_2]
        csvwriter.writerow(row)


with open("../../s1_rel.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    
    # Write Sheet 1 data
    for i in range(len(s1_df)):
        image_1 = s1_df.loc[i]["IMAGE1"].split("/")[-1] if type(s1_df.loc[i]["IMAGE1"]) != float else 0
        image_2 = s1_df.loc[i]["IMAGE2"].split("/")[-1] if type(s1_df.loc[i]["IMAGE2"]) != float else 0
        plate_num = s1_df.loc[i]["PLATE_READ"]
        row = [plate_num, image_1, image_2]
        csvwriter.writerow(row)
