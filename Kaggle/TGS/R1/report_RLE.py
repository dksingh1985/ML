from model import *
from data import *
import pandas as pd


PRED_MASK_PATH = "D:\\Data\\Kaggle\\TGS\\test\\mask"
#rle_df = pd.read_csv("D:\\Data\\Kaggle\\TGS\\sample_submission.csv")


msks = os.listdir(PRED_MASK_PATH)
rle_list = []
for i in range(len(msks)):
    if (i%100==0):
        print(">", i)
    temp = []
    temp.append(msks[i].split(".")[0])
    msknp = mskPredictNpy(PRED_MASK_PATH + "\\" + msks[i])
    rle = conv2RLE(msknp)
    temp.append(rle)
    if (len(rle.split()) % 2 == 1):
        print(">", i, "=", temp[0], ":" , len(rle.split()))
    
    rle_list.append(temp)
        
rle_df = pd.DataFrame(rle_list)
rle_df = rle_df.rename(index=str, columns={0: "id", 1: "rle_mask"})
rle_df = rle_df.sort_values(by='id', ascending=True)
rle_df.to_csv("D:\\Data\\Kaggle\\TGS\\submission.csv", header=True, index=False)


