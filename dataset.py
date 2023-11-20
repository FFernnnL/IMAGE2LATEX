import csv
import os
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

tokens_file = "D:/Code_conda/IMAGE2LATEX/im2latex100k/tokens.txt"  # 替换为你的标记文件路径
groundtruth_train = "D:/Code_conda/IMAGE2LATEX/im2latex100k/im2latex_train.csv"  # 替换为训练集的标注文件路径
groundtruth_val = "D:/Code_conda/IMAGE2LATEX/im2latex100k/im2latex_validate.csv"  # 替换为验证集的标注文件路径
groundtruth_test = "D:/Code_conda/IMAGE2LATEX/im2latex100k/im2latex_test.csv"  # 替换为测试集的标注文件路径
image_folder = "D:/Code_conda/IMAGE2LATEX/im2latex100k/formula_images_processed/"  # 替换为图像文件夹路径

# There are so many symbols (mostly escape sequences) that are in the test sets but not
# in the training set.
def remove_unknown_tokens(truth):
    """
    好像可以删掉（？）
    :param truth:
    :return:
    """
    # Remove \mathrm and \vtop are only present in the test sets, but not in the
    # training set. They are purely for formatting anyway.
    remaining_truth = truth.replace("\\mathrm", "")
    remaining_truth = remaining_truth.replace("\\vtop", "")
    # \; \! are spaces and only present in 2014's test set
    remaining_truth = remaining_truth.replace("\\;", " ")
    remaining_truth = remaining_truth.replace("\\!", " ")
    remaining_truth = remaining_truth.replace("\\ ", " ")
    # There's one occurrence of \dots in the 2013 test set, but it wasn't present in the
    # training set. It's either \ldots or \cdots in math mode, which are essentially
    # equivalent.
    remaining_truth = remaining_truth.replace("\\dots", "\\ldots")
    # Again, \lbrack and \rbrack where not present in the training set, but they render
    # similar to \left[ and \right] respectively.
    remaining_truth = remaining_truth.replace("\\lbrack", "\\left[")
    remaining_truth = remaining_truth.replace("\\rbrack", "\\right]")
    # Same story, where \mbox = \leavemode\hbox
    remaining_truth = remaining_truth.replace("\\hbox", "\\mbox")
    # There is no reason to use \lt or \gt instead of < and > in math mode. But the
    # training set does. They are not even LaTeX control sequences but are used in
    # MathJax (to prevent code injection).
    remaining_truth = remaining_truth.replace("<", "\\lt")
    remaining_truth = remaining_truth.replace(">", "\\gt")
    # \parallel renders to two vertical bars
    remaining_truth = remaining_truth.replace("\\parallel", "||")
    # Some capital letters are not in the training set...
    remaining_truth = remaining_truth.replace("O", "o")
    remaining_truth = remaining_truth.replace("W", "w")
    remaining_truth = remaining_truth.replace("\\Pi", "\\pi")
    return remaining_truth


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):
    truth_tokens = []
    remaining_truth = remove_unknown_tokens(truth).strip()
    while len(remaining_truth) > 0:
        try:
            matching_starts = [
                [i, len(tok)]
                for tok, i in token_to_id.items()
                if remaining_truth.startswith(tok)
            ]
            # Take the longest match
            index, tok_len = max(matching_starts, key=lambda match: match[1])
            truth_tokens.append(index)
            remaining_truth = remaining_truth[tok_len:].lstrip()
        except ValueError:
            raise Exception("Truth contains unknown token")
    return truth_tokens


def load_vocab(tokens_file):
    with open(tokens_file, "r") as fd:
        tokens = fd.read().splitlines()
        tokens.extend(SPECIAL_TOKENS)
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        id_to_token = {i: tok for i, tok in enumerate(tokens)}
        return token_to_id, id_to_token


def collate_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["formula"] for d in data],
            "encoded": torch.tensor(padded_encoded),
        },
    }

class CrohmeDataset(Dataset):
    """Dataset CROHME's handwritten mathematical formulas"""

    def __init__(
        self,
        groundtruth,
        tokens_file,
        root=None,
        ext=".png",
        crop=False,
        transform=None,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TSV file
            tokens_file (string): Path to tokens text file
            root (string): Path of the root directory of the dataset
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(CrohmeDataset, self).__init__()
        if root is None:
            root = os.path.dirname(groundtruth)
        self.crop = crop
        self.transform = transform
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        with open(groundtruth, "r") as fd:
            reader = csv.reader(fd)
            next(reader)  # 跳过CSV文件的标题行
            self.data = [
                {
                    "path": os.path.join(root, image_folder, row[1]),
                    "truth": {
                        "text": row[0],
                        "encoded": [
                            self.token_to_id[START],
                            *encode_truth(row[0], self.token_to_id),
                            self.token_to_id[END],
                        ],
                    },
                }
                for row in reader
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        # Remove alpha channel
        image = image.convert("RGB")

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)

        return {"path": item["path"], "truth": item["truth"], "image": image}
