# auto-diff-assignment
To run benchmark.py on CPU, pass `THEANO_FLAGS=device=cpu,floatX=float32`. For example:

```sh
THEANO_FLAGS=device=cpu,floatX=float32 python benchmark.py --model LSTM --learning_rate 0.01
```
