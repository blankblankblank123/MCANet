from __future__ import absolute_import
from .mcanet import *
from .resnet import *
from .msc import *
def MSC_MCANet():
	return MSC(
		base=MCANet_Infer(
			n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
		),
		scales=[0.5, 0.75,1.0],
	)

