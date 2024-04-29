import shutil
import os
import json
from difflib import SequenceMatcher as SM


def parse_csv(file_handle):
    lines = file_handle.readlines()
    headers = lines[0]
    data = lines[1:]
    num_records = len(data)

    return (headers, data, num_records)

def check_for_special(label, pred, special_dict):
    special = ["BU", "TA", "VT", "PH", "HR", "MB", "CA", "PV", "TP", "SB", "SM", "ST", "SP", ".", "-", "+", "&", "AMB"]
    correct_pred = False

    if label[:3] == "AMB" and (len(pred)+3 == len(label)):
        special_dict["AMB"] += 1
        if pred == label[3:]:
            correct_pred = True
    elif label[:2] in special and (len(pred)+2 == len(label)):
        special_dict[label[:2]] += 1
        if pred == label[2:]:
            correct_pred = True
    elif label[0] == "&":
        special_dict[label[0]] += 1
        if pred == label[1:-1]:
            correct_pred = True
    elif "." in label:
        special_dict["."] += 1
        if label.replace(".", "") == pred:
            correct_pred = True
    elif "-" in label:
        special_dict["-"] += 1
        if label.replace("-", "") == pred:
            correct_pred = True
    elif "+" in label:
        special_dict["+"] += 1
        if label.replace("+", "") == pred:
            correct_pred = True
    
    return correct_pred


### Added checking:
# prepend: 
#   CA(plate type: AHNR) 
#   PV,TP,MWRT(on plate) (plate type: ATNA)
#   -(minus sign) in the plate number somewhere (plate type: IPASS, MO,)
#   SB(plate type: SBNR)
#   SM(plate type: SMNR)
#   ST: STATE(on plate)(plate type: STNS)
#   SP: (plate type: STNS)
#   contains a . (plate type: 56)
#   AMB (Plate type AMNR)
#   BU (plate type BUNR)

correct = 0
incorrect = 0
num_records = 0
different_length = 0
special = ["BU", "TA", "VT", "PH", "HR", "MB", "CA", "PV", "TP", "SB", "SM", "ST", "SP", ".", "-", "+", "&", "AMB"]
special_dict = {letters:0 for letters in special}
correct_list = []
incorrect_but_train = 0

correct_dist = {}
incorrect_dist = {}
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.+&"
for char in characters:
    correct_dist[char] = 0
    incorrect_dist[char] = 0

pred_path = r"C:\Users\Jed\Desktop\capstone_project\4-25-Resnet50-1M-Small-Test-IPROC.csv"
all_train_test_path = os.path.join(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST", "all-train-test.txt")
with open(pred_path, "r") as file, open("incorrect.csv", "w") as incorrect_file, open("../CORRECT_LIST.csv", "w") as correct_file, open(all_train_test_path, "w") as tt_file:
    incorrect_file.write(f"LABEL,PRED,IMAGE\n")
    correct_file.write(f"PRED,IMAGE\n")
    tt_file.write(f"LABEL,PRED,IMAGE,IS_SPECIAL\n")
    headers, data, num_records = parse_csv(file)

    for line in data:
        split_line = line.strip().split(",")
        label,PRED_REG,PRED_INV,PRED_HE,PRED_INV_HE,image = split_line
        # label, pred, image = split_line
        pred = PRED_INV_HE
        correct_pred = False
        is_special = "False"
        
        if check_for_special(label, pred, special_dict):
            correct_pred = True
            correct += 1
            is_special = "True"
        else:         
            if pred == label:
                correct += 1
                correct_pred = True
            else:
                percent_incorrect = SM(None, label, pred).ratio() * 100

                if (len(pred) == len(label)) and (percent_incorrect > 70):
                    tt_file.write(f"{label},{pred},{image},{is_special}\n")
                    incorrect_but_train += 1
                elif (len(pred) != len(label)):
                    different_length += 1
                    

        
        if correct_pred:
            for char in label:
                correct_dist[char] += 1
            correct_file.write(f"{pred},{image}\n")
            tt_file.write(f"{label},{pred},{image},{is_special}\n")
            # correct_list.append(image)
        else:
            incorrect_file.write(f"{label},{pred},{image}\n")
            for char in label:
                incorrect_dist[char] += 1
            incorrect += 1

print()
print(f"Percentage chars correct: {(correct / num_records) * 100 :0.4f}%")
print(f"{correct}/{num_records}")
print()
print("Incorrect: ", incorrect)
print("Different Length: ", different_length)
print("Incorrect but train: ", incorrect_but_train)

# for key, value in special_dict.items():
#     print(f"{key}: {value}")



# with open("correct-dist-4-16.json", "w") as c_file, open("incorrect-dist-4-16.json", "w") as i_file:
#     json.dump(correct_dist, c_file)
#     json.dump(incorrect_dist, i_file)
