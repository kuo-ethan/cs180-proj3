import json
from scipy.spatial import Delaunay
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import cv2

from utils import display_img_with_keypoints, pick_and_record_keypoints, compute_affine, homogeneous_coords, bilinear_interpolation, display_img

# ===== Part 0: Parameters =====
PICK_KEYPOINTS = True
img1_name = 'ethan'
img2_name = 'shai'

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

display_img_with_keypoints(img1, img1_keypoints)
display_img_with_keypoints(img2, img2_keypoints)

# ===== Part 2: Computing the "Mid-way Face" =====
# Compute and display triangulation of average shape
avg_keypoints = (img1_keypoints + img2_keypoints) / 2
triangulation = Delaunay(avg_keypoints)
plt.figure()
plt.triplot(avg_keypoints[:, 1], avg_keypoints[:, 0], triangulation.simplices, color='blue')
plt.scatter(avg_keypoints[:, 1], avg_keypoints[:, 0], color='red')
plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()

morphed_img1 = np.zeros_like(img1)
morphed_img2 = np.zeros_like(img2)
midway_img = np.zeros_like(img2)
for i, j, k in triangulation.simplices:
    # Compute affine transformation matrix for both images (from average to source image)
    avg_tri = np.array((avg_keypoints[i], avg_keypoints[j], avg_keypoints[k]))
    img1_tri = np.array((img1_keypoints[i], img1_keypoints[j], img1_keypoints[k]))
    img2_tri = np.array((img2_keypoints[i], img2_keypoints[j], img2_keypoints[k]))
    A = compute_affine(avg_tri, img1_tri)
    B = compute_affine(avg_tri, img2_tri)
    # print('Affine matrix for img1:')
    # print(A)
    # print('Affine matrix for img2:')
    # print(B)

    # Get the coordinates inside the triangle in the average image
    # mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
    # cv2.fillConvexPoly(mask, np.array(avg_tri, dtype=np.int32), 1)
    # avg_tri_coords = np.nonzero(mask)
    # avg_tri_coords = np.array(list(zip(avg_tri_coords[0], avg_tri_coords[1])))
    row_coords, col_coords = polygon(avg_tri[:, 0], avg_tri[:, 1])
    avg_tri_coords = np.array(list(zip(row_coords, col_coords)))

    # Apply the transformation matrices to the average triangle
    avg_tri_coords_matrix = np.column_stack(homogeneous_coords(avg_tri_coords))
    img1_preimage_coords_matrix = A @ avg_tri_coords_matrix
    img2_preimage_coords_matrix = B @ avg_tri_coords_matrix

    # print('avg_tri_coords_matrix.shape: ', avg_tri_coords_matrix.shape)
    # print('img1_preimage_coords_matrix.shape: ', img1_preimage_coords_matrix.shape)
    # print('img2_preimage_coords_matrix.shape: ', img2_preimage_coords_matrix.shape)

    # Sample colors for each preimage using bilinear interpolation
    img1_preimage_coords = img1_preimage_coords_matrix[:2].T # unpack the raw coordiantes from homogeneous matrix
    img2_preimage_coords = img2_preimage_coords_matrix[:2].T
    # print('img1_preimage_coords: ', img1_preimage_coords.shape)
    # print('img2_preimage_coords: ', img2_preimage_coords.shape)
    img1_preimage_colors = bilinear_interpolation(img1, img1_preimage_coords)
    img2_preimage_colors = bilinear_interpolation(img2, img2_preimage_coords)
    # print('img1_preimage_colors', img1_preimage_colors[0])

    # Assign averaged colors to the average triangle
    averaged_colors = (img1_preimage_colors + img2_preimage_colors) / 2
    for idx, (r, c) in enumerate(avg_tri_coords):
        morphed_img1[r,c] = img1_preimage_colors[idx]
        morphed_img2[r,c] = img2_preimage_colors[idx]
        midway_img[r,c] = averaged_colors[idx]

# Display midway image
display_img(morphed_img1)
display_img(morphed_img2)
display_img(midway_img)

# Save midway image
skio.imsave(f'../images/{img1_name}_to_{img2_name}.jpg', midway_img)