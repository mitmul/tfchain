# rf-chain

Alternative Chain implementation for TensorFlow backend

# Requirements

- CUDA 8.0
- cuDNN 5.1
- Chainer 1.17.0+
- TensorFlow 0.11.0rc2

## Environmental Setup

```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl
pip install --upgrade -I setuptools
pip install --upgrade $TF_BINARY_URL
```

# Run tests

```
nosetests -s tests
```
