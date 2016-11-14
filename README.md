rf-chain
========

# Requirements

- CUDA 8.0
- cuDNN 5.1
- Chainer 1.17.0+
- TensorFlow 0.11.0 rc2

## Environmental Setup

```
conda create -n tensorflow python=3.5
source activate tensorflow
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl
pip install --upgrade -I setuptools
pip install --upgrade $TF_BINARY_URL
```

When the environment doesn't switch to the created one, use:

```
source ~/.pyenv/versions/anaconda3-4.1.1/bin/activate tensorflow
```

instead of `source activate tensorflow`.

To deactivate the environmen, use:

```
source deactivate tensorflow
```

##
