'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-19 21:43:35
LastEditors: Jiachen Sun
LastEditTime: 2021-07-19 22:09:41
'''

#TODO wip


import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.model_ae_conv_32x32x32_bin import AutoencoderConv

def train(args):
    model = AutoencoderConv().cuda()
    if args.load:
        model.load_state_dict(torch.load(args.chkpt))
        print("Loaded model from", args.chkpt)
    model.train()
    print("Done setup model")

    mse_loss = nn.MSELoss()
    adam = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    sgd = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer = adam

    for ei in range(args.resume_epoch, args.num_epochs):
        for bi, (img, patches, _) in enumerate(dataloader):

            avg_loss = 0
            for i in range(6):
                for j in range(10):
                    x = Variable(patches[:, :, i, j, :, :]).cuda()
                    y = model(x)
                    loss = mse_loss(y, x)

                    avg_loss += (1/60) * loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print('[%3d/%3d][%5d/%5d] loss: %f' % (ei, args.num_epochs, bi, len(dataloader), avg_loss))

            # save img
            if bi % args.out_every == 0:
                out = torch.zeros(6, 10, 3, 128, 128)
                for i in range(6):
                    for j in range(10):
                        x = Variable(patches[0, :, i, j, :, :].unsqueeze(0)).cuda()
                        out[i, j] = model(x).cpu().data

                out = np.transpose(out, (0, 3, 1, 4, 2))
                out = np.reshape(out, (768, 1280, 3))
                out = np.transpose(out, (2, 0, 1))

                y = torch.cat((img[0], out), dim=2).unsqueeze(0)
                save_imgs(imgs=y, to_size=(3, 768, 2 * 1280), name=f"out/{args.exp_name}/out_{ei}_{bi}.png")

            # save model
            if bi % args.save_every == args.save_every - 1:
                torch.save(model.state_dict(), f"checkpoints/{args.exp_name}/model_{ei}_{bi}.state")

    torch.save(model.state_dict(), f"checkpoints/{args.exp_name}/model_final.state")