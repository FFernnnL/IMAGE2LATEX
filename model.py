from dataprocess_printed import DataProcessP
from dataprocess_handwriting import DataProcessH
import pickle

with open("train_set.pkl", "rb") as f:
    train_set = pickle.load(f)