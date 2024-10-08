import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import json
import skimage.io as skio
from scipy.spatial import Delaunay
from skimage.draw import polygon
from PIL import Image


def pick_and_record_keypoints(img1, img2, json_path):
    # Create empty lists to store the points for each image
    points_img1 = []
    points_img2 = []
    
    # Flag to know when to stop
    picking = [True]

    # Function to handle clicks on image 1
    def onclick_img1(event):
        if event.inaxes == ax1 and picking[0]:
            x, y = event.xdata, event.ydata
            points_img1.append((y, x))
            ax1.scatter(x, y, color='red', marker='o')
            plt.draw()

    # Function to handle clicks on image 2
    def onclick_img2(event):
        if event.inaxes == ax2 and picking[0]:
            x, y = event.xdata, event.ydata
            points_img2.append((y, x))
            ax2.scatter(x, y, color='red', marker='o')
            plt.draw()

    # Function to stop picking points
    def stop_picking(event):
        picking[0] = False
        plt.close()

    # Display the two images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Show images on the axes
    ax1.imshow(img1)
    ax1.set_title('Image 1')
    ax2.imshow(img2)
    ax2.set_title('Image 2')

    # Add a "Stop" button to stop picking points
    stop_ax = plt.axes([0.45, 0.01, 0.1, 0.05])  # Position of the button
    stop_button = Button(stop_ax, 'Stop')
    stop_button.on_clicked(stop_picking)

    # Connect the click event to the correct functions
    fig.canvas.mpl_connect('button_press_event', onclick_img1)
    fig.canvas.mpl_connect('button_press_event', onclick_img2)

    # Display the images and interact
    plt.show()

    # Save the points as a JSON
    data = {
        "image1_keypoints": points_img1,
        "image2_keypoints": points_img2
    }

    # Save the dictionary as a JSON file
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # indent=4 makes the JSON output more readable

    # Return the points after the user is done
    return points_img1, points_img2

def display_img_with_keypoints(img, shape):
    plt.imshow(img)
    y_coords, x_coords = zip(*shape)
    plt.scatter(x_coords, y_coords, color='red', marker='o', s=50)
    plt.show()

def display_img(img):
    plt.imshow(img)
    plt.show()

def display_triangulation(keypoints, triangulation):
    plt.figure()
    plt.triplot(keypoints[:, 1], keypoints[:, 0], triangulation.simplices, color='blue')
    plt.scatter(keypoints[:, 1], keypoints[:, 0], color='red')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()


# Compute an affine transformation matrix between 2 triangles
def compute_affine(tri1, tri2):
    p1_x, p1_y = tri1[0][0], tri1[0][1]
    p2_x, p2_y = tri1[1][0], tri1[1][1]
    p3_x, p3_y = tri1[2][0], tri1[2][1]

    q1_x, q1_y = tri2[0][0], tri2[0][1]
    q2_x, q2_y = tri2[1][0], tri2[1][1]
    q3_x, q3_y = tri2[2][0], tri2[2][1]

    vector = np.array([q1_x, q1_y, q2_x, q2_y, q3_x, q3_y])
    matrix = np.array([[p1_x, p1_y, 1, 0, 0, 0],
                       [0, 0, 0, p1_x, p1_y, 1],
                       [p2_x, p2_y, 1, 0, 0, 0],
                       [0, 0, 0, p2_x, p2_y, 1],
                       [p3_x, p3_y, 1, 0, 0, 0],
                       [0, 0, 0, p3_x, p3_y, 1]])
    
    unknowns = np.linalg.inv(matrix) @ vector
    a, b, c, d, e, f = unknowns
    A = np.array([[a, b, c],
                 [d, e, f],
                 [0, 0, 1]])

    return A


# Appends a one to each coordinate
def homogeneous_coords(coords):
    ones = np.ones((coords.shape[0], 1))
    return np.hstack([coords, ones])


# Samples the color for each coordinate using bilinear interpolation on the image
def bilinear_interpolation(img, coords):
    height, width = img.shape[:2]

    # Create the grid for the image
    x = np.arange(width)
    y = np.arange(height)

    # For each color channel (R, G, B), perform bilinear interpolation
    interpolated_colors = []
    for c in range(3):
        interpolator = RegularGridInterpolator((y, x), img[:, :, c], bounds_error=False, fill_value=None)
        interpolated_colors.append(interpolator(coords))

    # Stack the interpolated results for R, G, B channels into shape (n, 3)
    interpolated_colors = np.stack(interpolated_colors, axis=-1)

    return interpolated_colors


# Morph an image into a new shape using a given triangulation
def morph_into(src_name, src_im, src_pts, dst_name, dst_pts, tri, save_morphed=False):
    dst_im = np.zeros_like(src_im)
    for i, j, k in tri.simplices:
        dst_tri = np.array((dst_pts[i], dst_pts[j], dst_pts[k]))
        src_tri = np.array((src_pts[i], src_pts[j], src_pts[k]))

        # Get the coordinates in the destination image we are colorizing
        row_coords, col_coords = polygon(dst_tri[:, 0], dst_tri[:, 1])
        dst_tri_coords = np.array(list(zip(row_coords, col_coords)))

        # Edge case when triangle is so "slim" it contains no pixels
        if len(dst_tri_coords) == 0:
            continue

        # Compute inverse affine transformation matrix
        A = compute_affine(dst_tri, src_tri)

        # Apply the transformation matrices to the average triangle
        dst_tri_coords_matrix = np.column_stack(homogeneous_coords(dst_tri_coords))
        src_coords_matrix = A @ dst_tri_coords_matrix

        # Sample colors for the preimage using bilinear interpolation
        src_coords = src_coords_matrix[:2].T # unpack the raw coordiantes from homogeneous matrix
        src_colors = bilinear_interpolation(src_im, src_coords)

        # Assign averaged colors to the average triangle
        for idx, (r, c) in enumerate(dst_tri_coords):
            dst_im[r,c] = src_colors[idx]
    
    if save_morphed:
        skio.imsave(f'../images/{src_name}_morphed_into_{dst_name}.jpg', dst_im)
    
    return dst_im

# Given 2 corresponding images and a triangulation, compute an intermediate morphed image between the 2
def compute_intermediate(im1_name, im2_name, im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac, save_intermediate=False, save_morphed=False):
    assert im1.shape == im2.shape
    assert len(im1_pts) == len(im2_pts)

    # Compute the intermediate shape based on the warp fraction
    intermediate_pts = im1_pts * (1-warp_frac) + im2_pts * warp_frac
    intermediate_identifier = f'{im1_name}-{im2_name}-warped[{warp_frac}]'

    # Morph both images into the intermediate shape
    morphed_im1 = morph_into(im1_name, im1, im1_pts, intermediate_identifier, intermediate_pts, tri, save_morphed)
    morphed_im2 = morph_into(im2_name, im2, im2_pts, intermediate_identifier, intermediate_pts, tri, save_morphed)

    # Cross dissolve the morphed images to add color
    intermediate_im = morphed_im1.astype(np.float32) * (1-dissolve_frac) + morphed_im2.astype(np.float32) * dissolve_frac
    if save_intermediate:
        skio.imsave(f'../images/{im1_name}_{im2_name}[{warp_frac}:{dissolve_frac}]_intermediate.jpg', intermediate_im)
    return intermediate_im.astype(np.uint8)


# Given 2 corresponding images and a triangulation, compute an intermediate morphed image between the 2
# def compute_intermediate(im1_name, im2_name, im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac, save_intermediate=False, save_morphed=False):
#     assert im1.shape == im2.shape
#     assert len(im1_pts) == len(im2_pts)
#     morphed_img1 = np.zeros_like(im1)
#     morphed_img2 = np.zeros_like(im1)
#     avg_img = np.zeros_like(im1)

#     # Compute the intermediate shape based on the warp fraction
#     avg_pts = im2_pts * warp_frac + im1_pts * (1-warp_frac)

#     for i, j, k in tri.simplices:
#         avg_tri = np.array((avg_pts[i], avg_pts[j], avg_pts[k]))
#         img1_tri = np.array((im1_pts[i], im1_pts[j], im1_pts[k]))
#         img2_tri = np.array((im2_pts[i], im2_pts[j], im2_pts[k]))

#         # Get the coordinates inside the triangle in the average image
#         row_coords, col_coords = polygon(avg_tri[:, 0], avg_tri[:, 1])
#         avg_tri_coords = np.array(list(zip(row_coords, col_coords)))


#         # Edge case when triangle is so "slim" it contains no pixels
#         if len(avg_tri_coords) == 0:
#             continue

#         # Compute affine transformation matrix for both images (from intermediate to source image)
#         A = compute_affine(avg_tri, img1_tri)
#         B = compute_affine(avg_tri, img2_tri)

#         # Apply the transformation matrices to the average triangle
#         avg_tri_coords_matrix = np.column_stack(homogeneous_coords(avg_tri_coords))
#         img1_preimage_coords_matrix = A @ avg_tri_coords_matrix
#         img2_preimage_coords_matrix = B @ avg_tri_coords_matrix

#         # Sample colors for each preimage using bilinear interpolation
#         img1_preimage_coords = img1_preimage_coords_matrix[:2].T # unpack the raw coordiantes from homogeneous matrix
#         img2_preimage_coords = img2_preimage_coords_matrix[:2].T
#         img1_preimage_colors = bilinear_interpolation(im1, img1_preimage_coords)
#         img2_preimage_colors = bilinear_interpolation(im2, img2_preimage_coords)

#         # Assign averaged colors to the average triangle
#         averaged_colors = img2_preimage_colors * dissolve_frac + img1_preimage_colors * (1-dissolve_frac)
#         for idx, (r, c) in enumerate(avg_tri_coords):
#             morphed_img1[r,c] = img1_preimage_colors[idx]
#             morphed_img2[r,c] = img2_preimage_colors[idx]
#             avg_img[r,c] = averaged_colors[idx]

#     if save_intermediate:
#         # Save midway image
#         skio.imsave(f'../images/{im1_name}_{im2_name}[{warp_frac}:{dissolve_frac}]_intermediate.jpg', avg_img)
    
#     if save_morphed:
#         # Save image1 and image2 morphed into the shape of the intermediate
#         skio.imsave(f'../images/{im1_name}_{im2_name}[{warp_frac}:{dissolve_frac}]_{im1_name}_intermediate.jpg', morphed_img1)
#         skio.imsave(f'../images/{im1_name}_{im2_name}[{warp_frac}:{dissolve_frac}]_{im2_name}_intermediate.jpg', morphed_img2)
    
#     return avg_img


def morph_multi(im_names, ims, corresponding_keypoints, warp_frac=1/45, dissolve_frac=1/45, frames=45):
    """
    Morph multiple images, step by step, each using a new triangulation. Generates a GIF of the morphing sequence.

    Parameters:
    -----------
    im_names : list of str
        A list of the names of the images being morphed.
    ims : list of np.array
        A list of image arrays to be morphed from the first to the last in sequence.
    corresponding_keypoints : list of tuple
        A list of corresponding keypoints between successive image pairs. Each element is a tuple of 
        (keypoints_im1, keypoints_im2) for each pair of consecutive images.
    save_intermediate : bool
        If True, saves intermediate frames of the morphing sequence.
    save_morphed : bool
        If True, saves the morphed version of each image into the averaged shape.
    warp_frac : float, optional
        The fraction of the warp applied at each morphing step (default is 1/45).
    dissolve_frac : float, optional
        The fraction of the dissolve applied at each morphing step (default is 1/45).
    frames : int, optional
        The number of frames to generate between each consecutive image pair (default is 45).

    Returns:
    --------
    None
    """
    n = len(ims)
    morph_sequence = []

    assert len(corresponding_keypoints) == n - 1

    for i in range(n-1):
        im1, im2 = ims[i], ims[i+1]
        im1_name, im2_name = im_names[i], im_names[i+1]
        im1_pts, im2_pts = corresponding_keypoints[i]
        avg_pts = (im1_pts + im2_pts) / 2
        triangulation = Delaunay(avg_pts)
        for t in range(frames+1):
            morph_sequence.append(compute_intermediate(
                im1_name, im2_name, im1, im2, im1_pts, im2_pts, triangulation, warp_frac * t, dissolve_frac * t
            ))

    im_names_str = "_to_" .join(im_names)
    save_gif(morph_sequence, f'../images/{im_names_str}.gif')



# Saves a GIF, morphing forward then backward
def save_gif(morph_sequence, output_path, duration=100):
    images = [Image.fromarray((image_array).astype(np.uint8)) for image_array in morph_sequence]
    images_reversed = images[::-1]
    images[0].save(output_path, save_all=True, append_images=images + images_reversed, duration=duration, loop=0)