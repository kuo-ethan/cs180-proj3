import json
from scipy.spatial import Delaunay
import skimage.io as skio
import numpy as np

from utils import display_img_with_keypoints, display_img, pick_and_record_keypoints, compute_intermediate, morph

# ===== Part 0: Parameters =====
PICK_KEYPOINTS = False
img1_name = 'tzuyu'
img2_name = 'minjoo'

# ===== Part 1: Defining Correspondences =====
# Define and display the shapes for each image
img1 = skio.imread(f'../data/{img1_name}.jpg')
img2 = skio.imread(f'../data/{img2_name}.jpg')
json_path = f'../data/{img1_name}_{img2_name}.json'

if PICK_KEYPOINTS:
    # Query user for keypoints and save them
    img1_keypoints, img2_keypoints = pick_and_record_keypoints(img1, img2, json_path)
    img1_keypoints, img2_keypoints = np.array(img1_keypoints), np.array(img2_keypoints)
else:
    # Reuse the last keypoints for this image pair
    with open(json_path, 'r') as f:
        data = json.load(f)
    img1_keypoints, img2_keypoints = np.array(data['image1_keypoints']), np.array(data['image2_keypoints'])

# display_img_with_keypoints(img1, img1_keypoints)
# display_img_with_keypoints(img2, img2_keypoints)

# ===== Part 2: Computing the "Mid-way Face" =====
# Compute triangulation of average shape -- this will be used throughout the morphing sequence
avg_keypoints = (img1_keypoints + img2_keypoints) / 2
triangulation = Delaunay(avg_keypoints)

midway_img = compute_intermediate(img1, img2, img1_keypoints, img2_keypoints, triangulation, 1/2, 1/2)
# display_img(midway_img)
# skio.imsave(f'../images/{img1_name}_to_{img2_name}.jpg', midway_img)

# ===== Part 3: The Morph Sequence =====
morph(img1_name, img2_name, img1, img2, img1_keypoints, img2_keypoints, triangulation)
