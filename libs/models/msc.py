import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75, 1]

    def initial_C_msg_passing(self,Input):
        return self.base.initial_C_msg_passing(Input)
    def S_msg_passing(self,FC_list,F_list):
        return self.base.S_msg_passing(FC_list,F_list)
    def progressive_C_msg_passing_aggregation(self,FCS_list,F_list,b_list):
        return self.base.progressive_C_msg_passing_aggregation(FCS_list,F_list,b_list)
    def progressive_C_msg_passing_distribution(self,Input_set,CPG):
        return self.base.progressive_C_msg_passing_distribution(Input_set,CPG,self.scales)
    def segmentation(self,FP_list):
        # Original
        logits = self.base.segmentation(FP_list[0][2])
        _, _, H, W = logits.shape

        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        M_list = []
        for FP_pyramid in FP_list:
            # Scaled
            logits_pyramid = []
            for FP in FP_pyramid:
                logits_pyramid.append(self.base.segmentation(FP).cpu())

            # Pixel-wise max
            logits_all = [interp(l) for l in logits_pyramid]
            logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
            M_list.append(logits_max)

        return M_list
