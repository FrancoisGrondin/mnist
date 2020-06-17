
# Author: Francois Grondin

import argparse
import math
import matplotlib.pyplot as plt
import progressbar
import time

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Classes

class Net(nn.Module):

	# Override the parent method
	def __init__(self):

		# Call the parent method
		super(Net, self).__init__()

		# First batch normalization layer
		self.bn1 = nn.BatchNorm2d(num_features=1)

		# First convolutional layer
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), bias=False)

		# Second batch normalization layer
		self.bn2 = nn.BatchNorm2d(num_features=32)

		# Second convolutional layer
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), bias=False)

		# Third batch normalization layer
		self.bn3 = nn.BatchNorm2d(num_features=32)

		# Third convolutional layer
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), bias=False)

		# Fully connected layer (expressed as a conv layer)
		self.fc1 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), bias=True)

		# Softmax (we use LogSoftMax for stability purpose)
		self.lsf = nn.LogSoftmax(dim=1)

    # Forward pass
	def forward(self, x):

		# Compute 1st layer (batchnorm + conv + relu + maxpool)
		# (N,1,28,28) > (N,32,13,13)
		x = F.max_pool2d(F.relu(self.conv1(self.bn1(x))), (2,2))

		# Compute 2nd layer (batchnorm + conv + relu + maxpool)
		# (N,32,13,13) > (N,32,5,5)
		x = F.max_pool2d(F.relu(self.conv2(self.bn2(x))), (2,2))

		# Compute 3rd layer (batchnorm + conv + relu + maxpool)
		# (N,32,5,5) > (N,32,1,1)
		x = F.max_pool2d(F.relu(self.conv2(self.bn3(x))), (2,2))

		# Classification layer (fully connectd + logsoftmax)
		# (N,32,1,1) > (N,10)
		x = self.lsf(torch.squeeze(self.fc1(x)))

		return x

# Functions

def display(batch, nCols=8):

	batch_size = batch[0].shape[0]
	nRows = int(math.ceil(batch_size/nCols))

	for i in range(0,batch_size):

		img = batch[0][i,0,:,:]
		lbl = batch[1][i]

		plt.subplot(nRows,nCols,i+1)
		plt.imshow(img, cmap='gray')
		plt.title("{}".format(lbl))
		plt.xticks([])
		plt.yticks([])

	plt.show()

def train(model, device, loader, optimizer):

	# Switch model in train mode
	model.train()

	# Loop for each batch in the dataset
	for batch in progressbar.progressbar(loader):

		# Load data for this batch
		image = batch[0]
		target = batch[1]

		# This is a new batch, so we reset all the gradients
		optimizer.zero_grad()

		# Then we compute the forward pass
		output = model(image)

		# And the loss function
		loss = F.nll_loss(output, target)

		# Backpropagate the loss in all the graph
		loss.backward()

		# Update all the parameters using grad descend
		optimizer.step()

def eval(model, device, loader):

	# Switch model in eval mode
	model.eval()

	# Count the number of correct classification
	correct = 0
	total = 0

	# Loop for each batch in the dataset
	for batch in progressbar.progressbar(loader):

		# Load data for this batch
		image = batch[0]
		target = batch[1]

		# Compute the forward pass
		output = model(image)

		# Check if output matches label
		digits = torch.argmax(output, dim=1)

		# Add correct classification
		correct += torch.sum(torch.eq(digits,target)).item()

		# Add total number of samples
		total += target.shape[0]

	# Compute accuracy
	accuracy = correct / total

	return accuracy

# Some parameters we will use, that we can pass as arguments when starting the script

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--view', type=bool, default=False, help='If True, display an example of the dataset and exit')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We want to be able to replicate results. To do so, we will define a fixed random seed
# to ensure we get the same parameters initialization and dataset shuffling from every run

torch.manual_seed(args.random_seed)

# Here we will load the dataset used for training and testing. This is provided with torchvision 
# and we can simply download it automatically. Once downloaded, it is stored and we don't need to 
# download it again. We will download MNIST and save it to the folder 'data'

train_dataset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# We will now use a data loader, which makes everything easier for batch processing and shuffling
# the dataset. Remember we set the random seed to a specific value, so we should get the same
# random shuffling from one run to the other.

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

# Let's just look at what we have in the first batch
# We will plot the batch in a matrix of pictures with 8 columns and
# the number of rows needed to cover the full batch
# The title of each image corresponds to the label

if args.view == True:
	display(batch=next(iter(train_loader)))
	exit()

# Load the model to train (and by default initialize with random parameters)
model = Net()

# Mesure the accuracy of the model with random parameters (should be bad!)
train_accuracy = eval(model=model, device=device, loader=train_loader)
test_accuracy = eval(model=model, device=device, loader=test_loader)

# Print result
print('Epoch %u - Train accuracy: %2.2f%%, Test accuracy: %2.2f%%\n' % (0, train_accuracy * 100.0, test_accuracy * 100.0))

# Set the optimizer to Adam (one of the most popular)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Loop for each epoch
for epoch in range(0, args.epochs):

	# Here we update the model weights by going through the entire training set
	train(model=model, device=device, loader=train_loader, optimizer=optimizer)
	
	# And we then mesure the accuracy of the model given
	train_accuracy = eval(model=model, device=device, loader=train_loader)
	test_accuracy = eval(model=model, device=device, loader=test_loader)

	# Print result
	print('Epoch %u - Train accuracy: %2.2f%%, Test accuracy: %2.2f%%\n' % (epoch+1, train_accuracy * 100.0, test_accuracy * 100.0))










