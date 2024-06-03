# TransE

Use OpenKE to train TransE model.

### Data

```
entity2id.txt
relation2id.txt
train2id.txt
valid2id.txt
test2id.txt
```

### Usage

```bash
# preprocess data
python data.py

# add constrains
cd drugdata
python n-n.py
cd ..

# train
python train.py

# test
python test.py
```