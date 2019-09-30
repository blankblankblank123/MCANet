import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import sys
import cv2
import torch
from torch.autograd import Variable
from libs.models import *
import torch.nn.functional as F
import glob
import random
from libs.utils import  vis
########################## loading model ###########################
model = MSC_MCANet()
state_dict = torch.load("MCA_res101_40000.pth", map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.eval()
model.cuda()
preds, gts = [], []

if not os.path.exists('./result/demo'):
	os.makedirs('./result/demo')
image_list = glob.glob('./demo_images/*')
random.shuffle(image_list)
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


CI_array = torch.cat(CI_list,dim = 0).squeeze()
CI_array = CI_array.numpy()
y_pred = KMeans(n_clusters=3).fit_predict(CI_array)


model_pca = PCA(n_components=2)
X_pca = model_pca.fit_transform(CI_array)




plt.figure()
plt.axis([-5,5,-4.5,4.5])
plt.tick_params(labelsize=23)

plt.scatter(X_pca[:, 0],X_pca[:, 1], c=np.array(y_pred)) #021   120
plt.savefig('./result/demo/cluster.jpg')

image_list1 = [image_list[i] for i in range(len(y_pred)) if y_pred[i] == 0]
image_list2 = [image_list[i] for i in range(len(y_pred)) if y_pred[i] == 1]
image_list3 = [image_list[i] for i in range(len(y_pred)) if y_pred[i] == 2]

print(len(image_list1),len(image_list2),len(image_list3))



image_list_list = [image_list1,image_list2,image_list3]

for dd,tmp_image_list in enumerate(image_list_list):
	############### image preprocess ################
	Input_set = []
	for i,one_image in enumerate(tmp_image_list):
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
		b_list_ = [b_list[i].numpy().reshape(-1)[0] for i in range(len(b_list))]

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


	##################visualization##################
	pd_vis_list = []
	for i,(im,pd) in enumerate(zip(tmp_image_list,pd_list)):
		im = cv2.imread(im).astype(np.float32)
		im = cv2.resize(im,(321,321))
		pd_vis = vis(im,pd,0.7)
		if b_list_[i] < 0.5:
			pd_vis = cv2.putText(pd_vis,str(b_list_[i])[:4],(40,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(50,50,200),3)
		else:
			pd_vis = cv2.putText(pd_vis,str(b_list_[i])[:4],(40,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(50,200,50),3)
		pd_vis_list.append(pd_vis)


	all_pd = np.concatenate(pd_vis_list,axis = 1)
	cv2.imwrite('./result/demo/' + str(dd) + '.png',all_pd)

