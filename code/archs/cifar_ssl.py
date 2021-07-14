'''
Description: 
Autor: Jiachen Sun
Date: 2021-07-13 17:53:51
LastEditors: Jiachen Sun
LastEditTime: 2021-07-13 18:21:38
'''
from torch import nn
import copy

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x):
		return self.head(self.ext(x))

def extractor_from_layer3(net):
	layers = [net.conv1, net.bn1, net.relu, net.layer1, net.layer2, net.layer3, net.avgpool, ViewFlatten()]
	return nn.Sequential(*layers)

def extractor_from_layer2(net):
	layers = [net.conv1, net.bn1, net.relu, net.layer1, net.layer2]
	return nn.Sequential(*layers)

def head_on_layer2(net, classes):
	head = copy.deepcopy([net.layer3, net.avgpool])
	head.append(ViewFlatten())
	head.append(nn.Linear(64, classes))
	return nn.Sequential(*head)