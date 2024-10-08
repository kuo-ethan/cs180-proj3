import json
from scipy.spatial import Delaunay
import skimage.io as skio
import numpy as np

from utils import pick_and_record_keypoints, morph_multi

# Main driver function for morph sequences. 
def morph_driver(im_names, pick_keypoints=False):
    n = len(im_names)
    ims = [skio.imread(f'../data/{name}.jpg') for name in im_names]
    corresponding_keypoints = [] # Contains pairs of keypoints lists for each pair of adjacent images, [(im1_keypoints, im2_keypoints), (im2_keypoints, im3_keypoints), ...]
    for i in range(1, n):
        im1, im2 = ims[i-1], ims[i]
        im1_name, im2_name = im_names[i-1], im_names[i]
        json_path = f'../data/{im1_name}_{im2_name}.json'
        if pick_keypoints:
            # Query user for keypoints and save them
            im1_keypoints, im2_keypoints = pick_and_record_keypoints(im1, im2, json_path)
            im1_keypoints, im2_keypoints = np.array(im1_keypoints), np.array(im2_keypoints)
        else:
            # Reuse the last keypoints for this image pair
            with open(json_path, 'r') as f:
                data = json.load(f)
            im1_keypoints, im2_keypoints = np.array(data['image1_keypoints']), np.array(data['image2_keypoints'])
        
        corresponding_keypoints.append((im1_keypoints, im2_keypoints))
    
    morph_multi(im_names, ims, corresponding_keypoints)

morph_driver(['ethan', 'shai'])
morph_driver(['tzuyu', 'minjoo'])
morph_driver(['lia', 'julie'])
morph_driver(['ethan_young', 'ethan_now'])
morph_driver(['ethan_kuo', 'ehong_kuo', 'dad_kuo', 'mom_kuo', 'ejean_kuo'])

def neutral_mean_face():
    # Compute the average shape and triangulate

    # Morph each image into the average shape

    # Save 2 examples (original, morphed)

    # Compute the mean face by averaging all the morphed faces

    # Warp my face into the average shape

    # Warp mean face into my shape

    pass

def happy_mean_face():
    # Compute the average shape

    # Morph each image into the average shape

    # Save 2 examples (original, morphed)

    # Compute and save the mean face by averaging all the morphed faces

    # Warp my face into the average shape

    # Warp mean face into my shape

    pass