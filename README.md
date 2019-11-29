# pyklopp

```bash
pyklopp build MaskedDeepFFN (1, 28, 28)  --layers 200 100 --base 'data/'
pyklopp build MaskedDeepDAN --structure xyz.adj --base 'data/'
pyklopp train data/myModel.pth --data mnist --epochs 50 --lr 0.01
```

```python
import pyklopp

builder = pyklopp.ModelBuilder()
```