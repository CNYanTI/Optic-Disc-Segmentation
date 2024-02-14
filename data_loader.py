# data loader
from __future__ import print_function, division
from skimage import io, transform
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms


#==========================dataset load==========================
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self, sample):

		imidx, image, label, labelEdge = sample['imidx'], sample['image'], sample['label'], sample['labelEdge']
		img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
		lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)
		lbE = transform.resize(labelEdge, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)

		return {'imidx': imidx, 'image': img, 'label': lbl, 'labelEdge': lbE}


class RandomCrop(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		imidx, image, label, labelEdge = sample['imidx'], sample['image'], sample['label'], sample['labelEdge']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]
			labelEdge = labelEdge[::-1]

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]
		labelEdge = labelEdge[top: top + new_h, left: left + new_w]

		return {'imidx': imidx, 'image': image, 'label': label, 'labelEdge': labelEdge}


class ToTensorV2(object):

	def __call__(self, sample):
		totensor = transforms.ToTensor()
		imidx, image, label, labelEdge = sample['imidx'], sample['image'], sample['label'], sample['labelEdge']
		_image = image.copy()
		_label = label.copy()
		_labelEdge = labelEdge.copy()
		img = totensor(_image)
		lbl = totensor(_label)
		lbE = totensor(_labelEdge)

		return {'imidx': imidx, 'image': img, 'label': lbl, 'labelEdge': lbE}


class ColorJitterT(object):

	def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
		self.brightness = brightness
		self.contrast = contrast
		self.saturation = saturation
		self.hue = hue

	def __call__(self, sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		coloradjust = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
		img = coloradjust(image)
		return {'imidx': imidx, 'image': img, 'label': label}


class HorizontalFlip(object):

	def __init__(self):
		self.percentage = np.random.choice([0, 1])
		# self.percentage = 1

	def __call__(self, sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		horizontal_flip = transforms.RandomHorizontalFlip(p=self.percentage)
		img = horizontal_flip(image)
		lbl = horizontal_flip(label)
		return {'imidx': imidx, 'image': img, 'label': lbl}


class VerticalFlip(object):

	def __init__(self):
		self.percentage = np.random.choice([0, 1])

	def __call__(self, sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		vertical_flip = transforms.RandomVerticalFlip(p=self.percentage)
		img = vertical_flip(image)
		lbl = vertical_flip(label)
		return {'imidx': imidx, 'image': img, 'label': lbl}


class SalObjDataset(Dataset):
	def __init__(self, img_name_list, lbl_name_list, lbE_name_list, transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.labelEdge_name_list = lbE_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):

		image = io.imread(self.image_name_list[idx])
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
			labelEdge_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx], as_gray=True)
			labelEdge_3 = io.imread(self.labelEdge_name_list[idx], as_gray=True)

		label = np.zeros(label_3.shape[0:2])
		labelEdge = np.zeros(labelEdge_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:, :, 0]
			labelEdge = labelEdge_3[:, :, 0]
		elif(2==len(label_3.shape)):
			label = label_3
			labelEdge = labelEdge_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:, :, np.newaxis]
			labelEdge = labelEdge[:, :, np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:, :, np.newaxis]
			label = label[:, :, np.newaxis]
			labelEdge = labelEdge[:, :, np.newaxis]

		sample = {'imidx': imidx, 'image': image, 'label': label, 'labelEdge': labelEdge}

		if self.transform:
			sample = self.transform(sample)

		return sample