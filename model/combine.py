# -*- coding: utf-8 -*-
from torchvision import transforms
import torch.nn as nn
from layers import bilateral_solver
from layers.summarizer import PGL_SUM
import object_discovery as tokencut
from layers import dino
import metric


class PGL_TC(nn.Module):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Encompasses TokenCut saliency partioning with PGL-SUM model"""
        # Initialize variables to None, to be safe
        self.config = config
        
        # Model creation                
        self.backbone = dino.ViTFeat(config.bb_url,
                                     config.bb_feat_dim,
                                     config.vit_arch,
                                     config.vit_feat,
                                     config.patch_size
                                     )
        
        self.summarizer = PGL_SUM(input_size=config.input_size,
                                  output_size=config.input_size,
                                  num_segments=config.n_segments,
                                  heads=config.heads,
                                  fusion=config.fusion,
                                  pos_enc=config.pos_enc
                                  ).to(config.device)

    def saliency_partition(self, frame_features):
        t, c, w, h = frame_features.shape
        patch_size = self.config.patch_size
        
        # Compute the new size of the video for matching multiples of patch_size
        new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
        feat_w, feat_h = new_w // patch_size, new_h // patch_size
        
        # Resize the video to the new size using bicubic interpolation in torchvision
        frames = transforms.Resize((new_h, new_w),
                                   interpolation=transforms.InterpolationMode.BICUBIC
                                   )(frame_features)

        features = self.backbone(frames)

        _, bipartition, eigvec = tokencut.ncut(features, [feat_h, feat_w], [patch_size, patch_size],
                                               [h,w], self.config.tau)
        
        if self.config.fine_parition is False:
            return bipartition, eigvec
    
        # _, binary_solver = bilateral_solver.bilateral_solver_output(img_pth, bipartition, sigma_spatial = args.sigma_spatial, sigma_luma = args.sigma_luma, sigma_chroma = args.sigma_chroma)
        # mask1 = torch.from_numpy(bipartition).cuda()
        # mask2 = torch.from_numpy(binary_solver).cuda()
        # if metric.IoU(mask1, mask2) < 0.5:
        #     binary_solver = binary_solver * -1
        
        # return bipartition, binary_solver
    
    def importance_score(self, frame_features):
        output, _ = self.model(frame_features.squeeze(0))
        return output
        
    def forward(self, frames):
        
    

if __name__ == '__main__':
    pass
