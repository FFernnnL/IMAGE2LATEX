from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import torchvision
from torchvision import transforms as tvt
import math
import os
import pickle


class DataProcessP(Dataset):
    def __init__(
        self, data_path, img_path, data_type: str
    ):
        super().__init__()
        assert data_type in ["train", "test", "validate"], "Not found data type"
        csv_path = data_path + f"/im2latex_{data_type}.csv"
        df = pd.read_csv(csv_path)
        df["image"] = df.image.map(lambda x: img_path + "/" + x)
        self.walker = df.to_dict("records")
        self.transform = tvt.Compose([tvt.Normalize((0.5), (0.5)),])

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        formula = item["formula"]
        image = torchvision.io.read_image(item["image"])
        image = image.to(dtype=torch.float)
        image /= image.max()
        image = self.transform(image)  # transform image to [-1, 1]
        return image, formula


test_set = DataProcessP(
        data_path="./data/im2latex100k",
        img_path="./data/im2latex100k/formula_images_processed",
        data_type="test",
    )

with open("./processed_data/processed_data_printed/test_set.pkl", "wb") as f:
    pickle.dump(test_set, f)

train_set = DataProcessP(
        data_path="./data/im2latex100k",
        img_path="./data/im2latex100k/formula_images_processed",
        data_type="train",
    )

with open("./processed_data/processed_data_printed/train_set.pkl", "wb") as f:
    pickle.dump(train_set, f)


val_set = DataProcessP(
        data_path="./data/im2latex100k",
        img_path="./data/im2latex100k/formula_images_processed",
        data_type="validate",
    )

with open("./processed_data/processed_data_printed/validate_set.pkl", "wb") as f:
    pickle.dump(val_set, f)



