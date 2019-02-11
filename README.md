# BERT_WSD

# 1. 입력데이터 형식
```frame_identification.py```

```trn, dev, tst = dataio.load_framenet_data(language)``` (line 46)

[
  [
    ['Your', 'contribution', 'to', 'Goodwill', 'will', 'mean', 'more', 'than', 'you', 'may', 'know', '.'], 
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', 'may.v', '_', '_'], 
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', 'Likelihood', '_', '_']
  ],
  ...
]

# 2. 추가 필요 데이터
- lu2idx
- sense2idx
- lusensemap

