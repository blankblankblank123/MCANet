import cv2
import os
import numpy as np
import torch
from torch.autograd import Variable
from libs.models import *
import torch.nn.functional as F
import pdb
import glob
from libs.utils import scores
from libs.utils import  vis
from PIL import Image
save_dir = './result/msrc_result' #dir to segmentation result
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

image_dir_list = glob.glob('/home1/whc/internet/Data/MSRC/*')

########################## loading model ###########################
model = MSC_MCANet()
state_dict = torch.load("MCA_res101_40000.pth", map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.eval()
model.cuda()

score_list = [] # list to store metric for each class
for one_dir in image_dir_list:
	one_class_name = one_dir.split('/')[-1]

	preds, gts = [], []
	

	image_list = glob.glob(one_dir +'/*.bmp')
	image_list.sort()

	anno_list = glob.glob(one_dir + '/GroundTruth/*')
	anno_list.sort()
	############### image preprocess ################
	Input_set = []
	for i,one_image in enumerate(image_list):
		img_ = cv2.imread(one_image).astype(np.float32)
		img_ = cv2.resize(img_,(321,321))

		img = img_.copy()

		img -= np.array([104.008,116.669,122.675])

		img = img.transpose(2,0,1)

		img = Variable(torch.Tensor(img)).unsqueeze(0)
		Input_set.append(img.cuda())

	############### initial channel message-passing ############
	with torch.no_grad():
		FC_list,F_list,CI_list = model.initial_C_msg_passing(Input_set)

	################## spatial message-passing #################
	with torch.no_grad():
		S_list,b_list,FCS_list = model.S_msg_passing(FC_list,F_list)

	#################progressive message-passing ###############
	with torch.no_grad():
		CPG = model.progressive_C_msg_passing_aggregation(FCS_list,F_list,b_list)
		FP_list = model.progressive_C_msg_passing_distribution(Input_set,CPG)

	################# segmentation ###################
	with torch.no_grad():	
		M_list = model.segmentation(FP_list)
	pd_list = []
	for logits in M_list:
		logits = F.interpolate(
		    logits, size=(321,321), mode="bilinear", align_corners=False
		)
		probs = F.softmax(logits, dim=1)
		pd = torch.argmax(probs, dim=1).squeeze()
		pd = pd.cpu().data.numpy()

		pd_list.append(pd)

	################# metric #########################
	for pd,gt in zip(pd_list,anno_list):
		gt = cv2.imread(gt).astype(np.float32)
		gt = cv2.resize(gt,(321,321))
		gt = gt[:,:,0]
		gt[gt < 128.] = 0
		gt[gt >=128.] = 1

		preds += list(pd)
		gts += list(gt)


	##################visualization##################
	pd_vis_list = []
	for i,(im,pd) in enumerate(zip(image_list,pd_list)):
		im = cv2.imread(im).astype(np.float32)
		im = cv2.resize(im,(321,321))
		pd_vis = vis(im,pd,0.7)

		pd_vis_list.append(pd_vis)


	all_pd = np.concatenate(pd_vis_list,axis = 1)
	cv2.imwrite(save_dir + '/' + one_class_name + '.png',all_pd)

	score = scores(gts, preds, n_class=2)
	print(one_class_name)
	print(score)
	score_list.append(score)

iou_list = [one['Class IoU'][1] for one in score_list]
print('mean jaccard: ' + str(sum(iou_list)/len(iou_list)))

pre_list = [one['Pixel Accuracy'] for one in score_list]
print('mean precision: ' + str(sum(pre_list)/len(pre_list)))

