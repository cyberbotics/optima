#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import math
import random

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from operator import itemgetter

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ## Load and process dataset
# ### Train Steerings & Speeds

steeringFile = open("../dataset/CW-files/steering.txt")  
revSteeringFile = open("../dataset/CC-files/steering.txt")
speedFile = open("../dataset/CW-files/speed.txt")
revSpeedFile = open("../dataset/CC-files/speed.txt")

steerings = steeringFile.readlines()
revSteerings = revSteeringFile.readlines()
speeds = speedFile.readlines()
revSpeeds = revSpeedFile.readlines()

formatedSteerings = []
formatedRevSteerings = []
formatedSpeeds = []
formatedRevSpeeds = []
for string in steerings:
    new_string = string.replace("\n", "")
    formatedSteerings.append(float(new_string))
for string in revSteerings:
    new_string = string.replace("\n", "")
    formatedRevSteerings.append(float(new_string))
for string in speeds:
    new_string = string.replace("\n", "")
    formatedSpeeds.append(float(new_string))
for string in revSpeeds:
    new_string = string.replace("\n", "")
    formatedRevSpeeds.append(float(new_string))

fig = plt.figure(figsize=(8,6), dpi = 130)
plt.xlabel("steerings")
plt.plot(formatedSteerings[0:2000])
plt.show(block = False)
# fig.savefig("foo.pdf", bbox_inches='tight')

# Remove oscillations in the steering value with a window average
N=81 # odd nb for simplicity
avgSteerings = np.convolve(formatedSteerings, np.ones(N)/N, mode='valid')
avgRevSteerings = np.convolve(formatedRevSteerings, np.ones(N)/N, mode='valid')

fig = plt.figure(figsize=(8,6), dpi = 130)
plt.xlabel("steerings")
plt.plot(avgSteerings[0:2000])
plt.show(block = False)
# fig.savefig("foo1.pdf", bbox_inches='tight')

idxFormatedSteerings = list(enumerate(avgSteerings, int((N-1)/2)))
idxFormatedRevSteerings = list(enumerate(avgRevSteerings,int((N-1)/2)))

idxFormatedSpeeds = list(enumerate(formatedSpeeds))
idxFormatedRevSpeeds = list(enumerate(formatedRevSpeeds))

# Only keep even indexes of steering and speed values
idxFormatedSteerings = idxFormatedSteerings[::2] 
idxFormatedRevSteerings = idxFormatedRevSteerings[::2] 
idxFormatedSpeeds = idxFormatedSpeeds[::2] 
idxFormatedRevSpeeds = idxFormatedRevSpeeds[::2] 

allSteerings = list(idxFormatedSteerings) + list(idxFormatedRevSteerings)
allSpeeds = idxFormatedSpeeds[int((N-1)/4):-int((N-1)/4)] + idxFormatedRevSpeeds[int(N/4):-int(N/4)]

fig = plt.figure(figsize=(8,6), dpi = 130)
plt.xlabel("steerings")
hist0 = plt.hist(list(map((lambda x: x[1]), allSteerings)), bins = 100)
plt.show(block = False)
# fig.savefig("steerings_no_filter.pdf", bbox_inches='tight')

# Filter steerings around 0 which are too numerous
filtered = list(filter(lambda x: x[1][1] <= 0.01 and x[1][1] >= -0.01, list(enumerate(allSteerings))))
reducedFilteredTuples = random.sample(list(filtered),int(len(filtered)*(1-0.35)))
indexes, _ = zip(*reducedFilteredTuples)

for ele in sorted(indexes, reverse = True):
    del allSteerings[ele]
    del allSpeeds[ele]

previousSplit = 0
for idx,steer in enumerate(allSteerings):
    if steer[0] < previousSplit:
        split = idx
    previousSplit = steer[0]

allSteerings[:split] = [(0, x) for x in allSteerings[:split]]
allSteerings[split:] = [(1, x) for x in allSteerings[split:]]

# Random order for training part
randomIdx = list(np.random.permutation(len(allSteerings)))
allSteerings = [allSteerings[i] for i in randomIdx]
allSpeeds = [allSpeeds[i] for i in randomIdx]

# Distribution of speed and steering values
fig = plt.figure(figsize=(8,6), dpi = 130)
plt.xlabel("steerings")
hist1 = plt.hist(list(map((lambda x: x[1][1]), allSteerings)), bins = 100)
plt.show(block = False)
# fig.savefig("foo2.pdf", bbox_inches='tight')
fig = plt.figure()
hist2 = plt.hist(list(map((lambda x: x[1]), allSpeeds)), bins = 100)
plt.show(block = False)

print(len(allSteerings))

# ### Images

# Import images and augment data by flipping each image around y axis
def import_img_batch_train(start, end):
    images = []
    
    for direction,x in allSteerings[start:end]:
        if(direction == 0):
            img_center = cv2.imread('../dataset/CW-images/center_' + str(x[0]) + '.png').astype(np.float32)
            flip_img_center = cv2.flip(img_center, 1)
            img_left = cv2.imread('../dataset/CW-images/left_' + str(x[0]) + '.png').astype(np.float32)
            flip_img_left = cv2.flip(img_left, 1)
            img_right = cv2.imread('../dataset/CW-images/right_' + str(x[0]) + '.png').astype(np.float32)
            flip_img_right = cv2.flip(img_right, 1)
        else:
            img_center = cv2.imread('../dataset/CC-images/center_' + str(x[0]) + '.png').astype(np.float32)
            flip_img_center = cv2.flip(img_center, 1)
            img_left = cv2.imread('../dataset/CC-images/left_' + str(x[0]) + '.png').astype(np.float32)
            flip_img_left = cv2.flip(img_left, 1)
            img_right = cv2.imread('../dataset/CC-images/right_' + str(x[0]) + '.png').astype(np.float32)
            flip_img_right = cv2.flip(img_right, 1)
        images.append(img_center)
        images.append(flip_img_center)
        images.append(img_left)
        images.append(flip_img_left)
        images.append(img_right)
        images.append(flip_img_right)
    return images

plt.show()
# Compute mean and standard deviation per mini-batch
ncum = list(range(0,len(allSteerings),50)) + [len(allSteerings)]
ncum = np.array(ncum)
n = ncum.copy()
n[1:] -= n[:-1].copy()
n = np.delete(n, 0)

means = []
stds = []

for i in tqdm(range(len(n))):
    train_input = torch.from_numpy(np.array(import_img_batch_train(ncum[i],ncum[i+1]))).permute(0, 3, 1, 2)
    means.append(torch.mean(train_input, dim=[0,2,3]))
    stds.append(torch.std(train_input, dim=[0,2,3]))
del train_input

# Compute general mean and std for complete dataset
total = sum(n)
sumx = []
sumx2 = []

for i in range(len(n)):
    sumx.append(means[i] * n[i])

for i in range(len(n)):
    sumx2.append(stds[i]**2 * (n[i]-1) + (sumx[i]**2/n[i])) 

mean = sum(sumx)/total
std = torch.sqrt((sum(sumx2)-sum(sumx)**2 /total) / (total-1))
print(mean,std)

# Duplicate labels for the three images (center, left, right)
train_labels = list(zip(list(map((lambda x: x[1][1]), allSteerings)),list(map((lambda x: x[1]), allSpeeds))))
train_labels = [x for pair in zip(train_labels,train_labels,train_labels) for x in pair]
train_target0 = []

# Add offset to left and right camera to push car to center of line
for idx, x in enumerate(train_labels):
    if(idx % 3 == 0):
        train_target0.append(x)
    elif(idx % 3 == 1):
        train_target0.append((x[0]+0.08, x[1]))
    elif(idx % 3 == 2):
        train_target0.append((x[0]-0.08, x[1]))

train_target0 = [x for pair in zip(train_target0,train_target0) for x in pair]
train_target = []

# Invert steering for flipped images
for idx, x in enumerate(train_target0):
    if(idx % 2 == 0):
        train_target.append(x)
    else:
        train_target.append((-x[0], x[1]))

# Normalize output labels
train_target = torch.from_numpy(np.array(train_target).astype(np.float32))
target_mean = torch.mean(train_target, dim=0)
target_std = torch.std(train_target, dim=0)
train_target[:,0].sub_(target_mean[0]).div_(target_std[0])
train_target[:,1].sub_(target_mean[1]).div_(target_std[1])
print(target_mean, target_std)

# ## Training

def train_model(dvc, model, train_target, nb_train_batches, train_batches_size, mu, std):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    for e in (range(0, 20)):
        sum_loss = 0
        print(e)
        model.train()
        for b in tqdm(range(0, nb_train_batches)):
            
            # Load batch
            input_batch = torch.from_numpy(np.array(import_img_batch_train(train_batches_size[b], train_batches_size[b+1]))).permute(0, 3, 1, 2)
            
            # Apply normalization
            for channel in range(3):
                input_batch[:,channel,:,:].sub_(mu[channel]).div_(std[channel])
                
            # Push to GPU  
            input_batch = input_batch.to(dvc) 
            model = model.to(dvc)
            train_target = train_target.to(dvc)
            
            # Compute forward prop and loss
            output = model(input_batch)
            loss = criterion(output, train_target[train_batches_size[b]*6:train_batches_size[b+1]*6])
            sum_loss += float(loss.item())
            
            # Apply backward prop and learning step
            model.zero_grad()
            loss.backward()
            optimizer.step()
                    
        print("train loss = " + str(sum_loss))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(19456, 500)
        self.fc2 = nn.Linear(500, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.elu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.elu(F.max_pool2d(self.conv3(x), kernel_size=2))
        x = F.elu(self.fc1(x.reshape(-1, 19456)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet()

train_model(device, model, train_target, len(n), ncum , mean, std)
torch.save(model.state_dict(), "offsetModel.pt")

# ## Write model parameters to txt

model = ConvNet()
model.load_state_dict(torch.load("offsetModel.pt"))
weights = []
bias = []
for idx,param in enumerate(model.parameters()):
    if(idx % 2 == 0):
        weights.append(param)
    else:
        bias.append(param)

bfile1 = open('../controllers/CNN_autonomous_car_cpu_fixed/model/biases.txt', 'a')
bfile2 = open('../controllers/CNN_autonomous_car_cpu_float/model/biases.txt', 'a')
bfile3 = open('../controllers/CNN_autonomous_car_fpga/model/biases.txt', 'a')

for x in bias:
    for y in x:
        #print(y.item())
        bfile1.write(str(y.item()) + "\n")
        bfile2.write(str(y.item()) + "\n")
        
bfile1.close()
bfile2.close()
bfile3.close()

wfile1 = open('../controllers/CNN_autonomous_car_cpu_fixed/model/weights.txt', 'a')
wfile2 = open('../controllers/CNN_autonomous_car_cpu_float/model/weights.txt', 'a')
wfile3 = open('../controllers/CNN_autonomous_car_fpga/model/weights.txt', 'a')

# CONV
for x in weights[0:3]:
    for y in x:
        for z in y:
            for w in z:
                for r in w:
                    #print(r.item())
                    wfile1.write(str(r.item()) + "\n")
                    wfile2.write(str(r.item()) + "\n")

# LINEAR
for x in weights[3:4]:
    for y in x:
        for z in y:
            #print(z.item())
            wfile1.write(str(z.item()) + "\n")
            wfile2.write(str(z.item()) + "\n")

for x in weights[4:5]:
    for y in x:
        for z in y:
            #print(z.item())
            wfile1.write(str(z.item()) + "\n")
            wfile2.write(str(z.item()) + "\n")
            
wfile1.close()
wfile2.close()
wfile3.close()

