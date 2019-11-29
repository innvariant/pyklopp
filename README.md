# pyklopp

```bash
# Initializing & Saving: mymodel.py
pyklopp init mymodel (1, 28, 28) --layers 200 100 --save 'mymodel1/'
pyklopp init mymodel --structure xyz.adj --save 'mymodel2/'

# Training
pyklopp train --init mymodel.py --data mnist --epochs 50 --lr 0.01
pyklopp train --load mymodel1/ --data mnist --epochs 50 --lr 0.01
```

```python
import pyklopp

builder = pyklopp.ModelBuilder()
```