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

def compute_intermediate(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    assert im1.shape == im2.shape
    morphed_img1 = np.zeros_like(im1)
    morphed_img2 = np.zeros_like(im1)
    avg_img = np.zeros_like(im1)

    # Compute the intermediate shape based on the warp fraction
    avg_pts = im2_pts * warp_frac + im1_pts * (1-warp_frac)

    for i, j, k in tri.simplices:
        # Compute affine transformation matrix for both images (from intermediate to source image)
        avg_tri = np.array((avg_pts[i], avg_pts[j], avg_pts[k]))
        img1_tri = np.array((im1_pts[i], im1_pts[j], im1_pts[k]))
        img2_tri = np.array((im2_pts[i], im2_pts[j], im2_pts[k]))
        A = compute_affine(avg_tri, img1_tri)
        B = compute_affine(avg_tri, img2_tri)

        # Get the coordinates inside the triangle in the average image
        row_coords, col_coords = polygon(avg_tri[:, 0], avg_tri[:, 1])
        avg_tri_coords = np.array(list(zip(row_coords, col_coords)))

        # Apply the transformation matrices to the average triangle
        avg_tri_coords_matrix = np.column_stack(homogeneous_coords(avg_tri_coords))
        img1_preimage_coords_matrix = A @ avg_tri_coords_matrix
        img2_preimage_coords_matrix = B @ avg_tri_coords_matrix

        # Sample colors for each preimage using bilinear interpolation
        img1_preimage_coords = img1_preimage_coords_matrix[:2].T # unpack the raw coordiantes from homogeneous matrix
        img2_preimage_coords = img2_preimage_coords_matrix[:2].T
        img1_preimage_colors = bilinear_interpolation(im1, img1_preimage_coords)
        img2_preimage_colors = bilinear_interpolation(im2, img2_preimage_coords)

        # Assign averaged colors to the average triangle
        averaged_colors = img2_preimage_colors * dissolve_frac + img1_preimage_colors * (1-dissolve_frac)
        for idx, (r, c) in enumerate(avg_tri_coords):
            morphed_img1[r,c] = img1_preimage_colors[idx]
            morphed_img2[r,c] = img2_preimage_colors[idx]
            avg_img[r,c] = averaged_colors[idx]
    
    return avg_img

    # Display midway image
    display_img(morphed_img1)
    display_img(morphed_img2)
    display_img(avg_img)

    # Save midway image
    skio.imsave(f'../images/{img1_name}_to_{img2_name}.jpg', midway_img)


# Morph im1 to im2 using the triangulation 
def morph(im1_name, im2_name, im1, im2, im1_pts, im2_pts, tri, warp_frac=1/45, dissolve_frac=1/45, frames=45):
    morph_sequence = []
    for t in range(frames+1):
        morph_sequence.append(compute_intermediate(im1, im2, im1_pts, im2_pts, tri, warp_frac * t, dissolve_frac * t))

    # Save a GIF of the morph sequence
    save_gif(morph_sequence, f'../images/{im1_name}_to_{im2_name}.gif')

def save_gif(morph_sequence, output_path, duration=100):
    images = [Image.fromarray((image_array).astype(np.uint8)) for image_array in morph_sequence]
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)