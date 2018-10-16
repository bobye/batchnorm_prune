This directory has the codes that reproduce the CIFAR-10 experiment in paper:
```
@inproceedings{ye2018rethinking,
  title={Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers},
  author={Ye, Jianbo and Lu, Xin and Lin, Zhe and Wang, James Z},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

# Prerequisites
 - python2
 - tensorflow 1.4 or above (GPU support is optional)

# How to run cifar10 experiments with batch normalization with ISTA

The following steps assume you are using `resnet20` as the baseline model. You can change to another 4-layer convolution model `cnn4` by editting the `inference` function in file `cifar10.py`:
```python
def inference(dt, images, is_training=True, is_compressed=False):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """

  slim = tf.contrib.slim
  with slim.arg_scope([slim.batch_norm],
                      is_training=is_training):      
  
    #squeeze = cnn4(dt, images, is_compressed)
    squeeze = resnet20(dt, images, is_compressed)
    
  return squeeze

```


- Start from a relative small L1 penality and moderate number of steps to create a startup checkpoint. Note that the default learning rate is 0.1. 
```bash
$ mkdir /tmp/cifar10_train_0
$ mkdir /tmp/cifar10_train_1
$ python cifar10_train.py --l1_penalty 0.001 --max_steps 30000 --checkpoint_version 0
```

  The warm-up model will be saved to checkpoint version 1. Checkout the raw accuracy of the saved model:
```bash
$ python cifar10_eval.py --checkpoint_version 1
```

- Raise up the L1 penality and set the resulting model to newer checkpoint
```bash
$ mkdir /tmp/cifar10_train_2
$ python cifar10_train.py --l1_penalty 0.005 --max_steps 50000 --checkpoint_version 1
...
2018-10-16 05:49:42.966512: step 32800, loss = 0.59, group lasso = 29.12, fake sparsity = 0.34, (5044.4 examples/sec; 0.025 sec/batch)
2018-10-16 05:49:45.419254: step 32900, loss = 0.39, group lasso = 29.62, fake sparsity = 0.34, (5096.3 examples/sec; 0.025 sec/batch)
2018-10-16 05:49:47.859760: step 33000, loss = 0.43, group lasso = 29.11, fake sparsity = 0.34, (5121.9 examples/sec; 0.024 sec/batch)
2018-10-16 05:49:50.371544: step 33100, loss = 0.42, group lasso = 29.35, fake sparsity = 0.34, (4976.5 examples/sec; 0.025 sec/batch)
2018-10-16 05:49:52.900149: step 33200, loss = 0.49, group lasso = 29.45, fake sparsity = 0.34, (4943.4 examples/sec; 0.025 sec/batch)
2018-10-16 05:49:55.447468: step 33300, loss = 0.47, group lasso = 30.14, fake sparsity = 0.34, (4907.1 examples/sec; 0.025 sec/batch)
...
```
Allow enough iterations until the "fake sparsity" plateaus. With `--l1_penalty 0.005`, resnet20 can be compressed to one with about 37% sparsity. The pruned model will be saved to checkpoint version 2.

  checkout the raw accuracy of the saved model (the actual numbers can be a bit different)
```bash
$ python cifar10_eval.py --checkpoint_version 2
...
total params: 177164
...
2018-10-16 05:59:15.028371: precision @ 1 = 0.9116
2018-10-16 05:59:15.028480: test loss = 0.2654

```

- Finally we have a compact model with moderately good precision ready for finetuning checkpoint version 2

```bash
$ python cifar10_train.py --checkpoint_version 2 --l1_penalty 0.0 --max_steps 50000 --learning_rate 0.01
```
  checkout the fine-tuned accuracy of the pruned model (checkpoint version 3):
```bash
$ python cifar10_eval.py --checkpoint_version 3
...
total params: 177164
...
2018-10-16 06:25:50.297370: precision @ 1 = 0.9060
2018-10-16 06:25:50.297444: test loss = 0.3259
```

Other experimental details can be found in the paper.

# FAQ

- Why the numbers do not exactly match the numbers in the paper?

  Because the tensorflow training is not determinisitic.

- Why fine-tuned accuracy is even lower than the raw accuracy right after pruning?

  It just sometimes happens :)

- How to select `weight_decay` and `l1_penalty` for different pruned versions of model?

  Fix `weight_decay` across different pruning stages. Start `l1_penalty` from a small value for warm-up, and gradually increase it until `fake sparsity` is non-zero. 

----
CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Part of the training codes are adapted from

https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/
