from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import torchvision
from torchvision import transforms as tvt
import math
import os
import pickle


class DataProcessH(Dataset):
    def __init__(
        self, data_path, img_path
    ):
        super().__init__()

        df = pd.read_csv(data_path,sep='\t', header=None, names=["Column1", "Column2"])

        df["Column1"] = df["Column1"].map(lambda x: img_path + "/" + x)
        self.walker = df.to_dict("records")
        self.transform = tvt.Compose([tvt.Normalize((0.5), (0.5)),])

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        image = torchvision.io.read_image(item["Column1"])
        image = image.to(dtype=torch.float)
        image /= image.max()
        image = self.transform(image)  # transform image to [-1, 1]
        formula = item["Column2"]
        return image, formula


train_set = DataProcessH(
        data_path="./data/Handwriting/gt_split/train.tsv",
        img_path="./data/Handwriting/train",
    )

with open("./processed_data/processed_data_handwriting/train_set.pkl", "wb") as f:
    pickle.dump(train_set, f)


val_set = DataProcessH(
        data_path="./data/Handwriting/gt_split/validation.tsv",
        img_path="./data/Handwriting/train",
    )

with open("./processed_data/processed_data_handwriting/validate_set.pkl", "wb") as f:
    pickle.dump(val_set, f)


test_set = DataProcessH(
        data_path="./data/Handwriting/groundtruth_2013.tsv",
        img_path="./data/Handwriting/test/2013",
    )

with open("./processed_data/processed_data_handwriting/test_set_2013.pkl", "wb") as f:
    pickle.dump(test_set, f)