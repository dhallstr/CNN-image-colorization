
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
from os import walk

def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]
	img_ab_rs = img_lab_rs[:,:,1:]

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]
	tens_rs_ab = torch.Tensor(np.moveaxis(img_ab_rs, 2, 0))[None,:,:,:]

	return (tens_orig_l, tens_rs_l, tens_rs_ab)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode, align_corners=True)
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

def get_files_in_dir(directory):
	f = []
	for (dirpath, dirnames, filenames) in walk(directory):
		f.extend([directory + "/" + f for f in filenames])
		break

	return f

def load_images(filenames, use_gpu=False):
	data = []
	for file in filenames:
		img = load_img(file)
		(tens_l_orig, tens_l_rs, tens_ab_rs) = preprocess_img(img, HW=(256,256))

		if(use_gpu):
			tens_l_rs = tens_l_rs.cuda()
        
		data.append((tens_l_rs.reshape(1, 256, 256), tens_ab_rs.reshape(2, 256, 256)))
        
	return data

def separate_io(test_data):
	inputs = [x for x, _ in test_data]
	outputs = [y for _, y in test_data]
	inputs = torch.cat(inputs).reshape(-1, 1, 256, 256)
	outputs = torch.cat(outputs).reshape(-1, 2, 256, 256)
	return inputs, outputs