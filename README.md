# IMAGE2LATEX
Math formula images recognition and convert to LaTex

适应了印刷体数据集  
修改了一些错误  
model.py中加了self-attention，但参数还需要调  
但是一跑就卡住^^  

Run in terminal:
```sh
python train.py --prefix "2023-11-20-" -n 200
```

数据集存放格式：
im2latex100k
│
├── formula_images_processed
│   │
│   ├──.png
│
├── im2latex_formulas.norm.csv
│
├── im2latex_train.csv
│
├── im2latex_test.csv
│
├── im2latex_validate.csv
│
└── tokens.txt
