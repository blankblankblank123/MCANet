import cv2
import numpy as np
def vis(img_,labels,alpha):

	pd = np.zeros((np.shape(img_)[0],np.shape(img_)[1],3))
	tmp = img_.copy()

	convert_pd = (255. - tmp) * alpha + tmp

	fg_mask = (labels == 1)
	bg_mask = (labels != 1)

	pd = fg_mask[:,:,None] * img_ + bg_mask[:,:,None] * convert_pd

	_,binary_out = cv2.threshold(((labels*255).astype(np.float32)),127,255,cv2.THRESH_BINARY)

	_,contours,hierarch=cv2.findContours(binary_out.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	pd=cv2.drawContours(pd,contours,-1,(0,0,255),5)
	return pd