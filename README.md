# Demo with MNIST

We will assume you already have Python 3 installed on your machine.

## Create the virtual environment

Create the environment

```
python3 -m venv /path/mnist
```

Active it

```
source /path/mnist/bin/activate
```

Download librairies

```
pip3 install -r requirements.txt
```

## Train the network

Start the following script. By default will train for 10 epochs.

```
python3 main.py
```

## Expected output

Here's the expected output when training on MNIST.

```
Epoch 0 - Train accuracy: 9.74%, Test accuracy: 9.82%
Epoch 1 - Train accuracy: 98.07%, Test accuracy: 97.83%
Epoch 2 - Train accuracy: 98.75%, Test accuracy: 98.28%
Epoch 3 - Train accuracy: 98.85%, Test accuracy: 98.27%
Epoch 4 - Train accuracy: 99.07%, Test accuracy: 98.28%
Epoch 5 - Train accuracy: 99.07%, Test accuracy: 98.23%
Epoch 6 - Train accuracy: 99.44%, Test accuracy: 98.61%
Epoch 7 - Train accuracy: 99.42%, Test accuracy: 98.60%
Epoch 8 - Train accuracy: 99.43%, Test accuracy: 98.47%
Epoch 9 - Train accuracy: 99.54%, Test accuracy: 98.63%
Epoch 10 - Train accuracy: 99.61%, Test accuracy: 98.61%
```