# k-anonymity & Deep Learning

## Dataset 
* Dataset used throughout the experiment: Adult dataset (used only 300 samples from the original dataset)
* 我利用 Mondrian 的 k-anonymity 演算法( `K=3` )，我設定的 Quasi-Identifier 為 [ `age` , `education-num` , `hours-per-week` ]，並且將資料集中的 `fnlwgt` 、 `capital-gain` 、 `capital-loss` 去掉，所以 total feature column 為 11 個，要預測的 target 為 `income`

## Model

```python
Model(
  (net): Sequential(
    (0): Linear(in_features=11, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=128, out_features=64, bias=True)
    (4): ReLU()
    (5): Linear(in_features=64, out_features=2, bias=True)
    (6): Softmax(dim=1)
  )
)
```

## Evaluation

### Dataset `without` Anonymity
| Category/Metrics   | Precision | Recall |
| -------- | :-------: |  :-------: |
| Income <= 50  | 84.62%    |  78.57%    |
| Income > 50 | 82.35%     |   87.50%     |

<image src="https://i.imgur.com/eHeBOmv.png" width=70%>

### Dataset `with` Anonymity
| Category/Metrics   | Precision | Recall |
| -------- | :-------: |  :-------: |
| Income <= 50  | 71.43%    |  66.67%    |
| Income > 50 | 68.75%    |   73.33%    |

<image src="https://imgur.com/zYgjiUd.png" width=70%>


### Different privacy level(k value) comparison
| Privacy Level(k value)/Metrics   | Accuracy | AUC |
| :-------- | :-------: |  :-------: |
| k=2 | 80%    |  0.85    |
| k=3 | 70%    |    0.76    |
| k=4 | 67%    |   0.72    |

<image src="https://i.imgur.com/RJDJO0x.png" width=70%>