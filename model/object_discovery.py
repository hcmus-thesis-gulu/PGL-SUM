import torch
import torch.nn.functional as F
import numpy as np
from torch.linalg import eigh
from scipy import ndimage
#from sklearn.mixture import GaussianMixture
#from sklearn.cluster import KMeans

def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    feats = F.normalize(feats, p=2, dim=0)
    adjacency_matrix = (feats.transpose(0,1) @ feats)
    # A = A.cpu().numpy()
    
    if no_binary_graph:
        adjacency_matrix[adjacency_matrix<tau] = eps
    else:
        adjacency_matrix = adjacency_matrix > tau
        adjacency_matrix = torch.where(adjacency_matrix.float() == 0, eps, adjacency_matrix)
    diagonal_sum = torch.sum(adjacency_matrix, dim=1)
    diagonal_matrix = torch.diag(diagonal_sum)

    # Print second and third smallest eigenvector
    _, eigenvectors = eigh(diagonal_matrix-adjacency_matrix, diagonal_matrix, eigvals=(1,2))
    eigenvec = eigenvectors[:, 0].clone()

    # method1 avg
    second_smallest_vec = eigenvectors[:, 0]
    avg = torch.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg

    seed = torch.argmax(torch.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = torch.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _, objects, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size) ## We only extract the principal object BBox
    mask = torch.zeros(dims)
    mask[cc[0],cc[1]] = 1

    mask = mask.to('cuda')
#    mask = torch.from_numpy(bipartition).to('cuda')
    bipartition = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
    

    eigvec = second_smallest_vec.reshape(dims) 
    eigvec = eigvec.to('cuda')
    eigvec = F.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
    return seed.cpu().numpy(), bipartition.cpu().numpy(), eigvec.cpu().numpy()

def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]


    if principle_object:
        mask = np.where(objects == cc)
       # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError
