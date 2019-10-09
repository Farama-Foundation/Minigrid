#!/usr/bin/env python3

import time
import random
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
import babyai

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
import cv2
import PIL

##############################################################################

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    #arr = torch.from_numpy(arr).float()
    arr = torch.from_numpy(arr)
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

def init_weights(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ImageBOWEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, reduce_fn=torch.mean):
        super(ImageBOWEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.reduce_fn = reduce_fn
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.long())
        embeddings = self.reduce_fn(embeddings, dim=1)
        embeddings = torch.transpose(embeddings, 1, 3)
        embeddings = torch.transpose(embeddings, 2, 3)
        return embeddings

class Flatten(nn.Module):
    """
    Flatten layer, to flatten convolutional layer output
    """

    def forward(self, input):
        return input.view(input.size(0), -1)

class Print(nn.Module):
    def forward(self, input):
        print(input.size())
        return input

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(

            #ImageBOWEmbedding(765, embedding_dim=16, padding_idx=0, reduce_fn=torch.mean),
            #nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1),
            #nn.LeakyReLU(),

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=6, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=6, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=6, stride=2),
            nn.LeakyReLU(),

            #Print(),
            Flatten(),

            nn.Linear(144, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

        self.apply(init_weights)

    def forward(self, obs):
        obs = obs / 16

        out = self.layers(obs)

        return out

    def present_prob(self, obs):
        obs = make_var(obs).unsqueeze(0)

        logits = self(obs)
        probs = F.softmax(logits, dim=-1)
        probs = probs.detach().cpu().squeeze().numpy()

        return probs[1]





env = gym.make('BabyAI-GoToRedBall-v0')

def sample_batch(batch_size=128):
    imgs = []
    labels = []

    for i in range(batch_size):
        obs = env.reset()['image']

        ball_visible = ('red', 'ball') in Grid.decode(obs)

        obs = env.get_obs_render(obs, tile_size=8, mode='rgb_array')
        obs = obs.transpose([2, 0, 1])

        imgs.append(np.copy(obs))
        labels.append(ball_visible)

    imgs = np.stack(imgs).astype(np.float32)
    labels = np.array(labels, dtype=np.long)

    return imgs, labels




print('Generating test set')
test_imgs, test_labels = sample_batch(256)

def eval_model(model):
    num_true = 0

    for idx in range(test_imgs.shape[0]):
        img = test_imgs[idx]
        label = test_labels[idx]

        p = model.present_prob(img)
        out_label = p > 0.5

        #print(out_label)

        if np.equal(out_label, label):
            num_true += 1
        #else:
        #    if label:
        #        print("incorrectly predicted as absent")
        #    else:
        #        print("incorrectly predicted as present")

    acc = 100 * (num_true / test_imgs.shape[0])
    return acc










##############################################################################

batch_size = 64

model = Model()
model.cuda()

optimizer = optim.Adam(
    model.parameters(),
    lr=5e-4
)

criterion = nn.CrossEntropyLoss()

running_loss = None

for batch_no in range(1, 10000):
    batch_imgs, labels = sample_batch(batch_size)
    batch_imgs = make_var(batch_imgs)
    labels = make_var(labels)

    pred = model(batch_imgs)

    loss = criterion(pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.data.detach().item()
    running_loss = loss if running_loss is None else 0.99 * running_loss + 0.01 * loss

    print('batch #{}, frames={}, loss={:.5f}'.format(
        batch_no,
        batch_no * batch_size,
        running_loss
    ))

    if batch_no % 25 == 0:
        acc = eval_model(model)
        print('accuracy: {:.2f}%'.format(acc))
