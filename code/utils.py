import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import json
import skimage.io as skio
from scipy.spatial import Delaunay
from skimage.draw import polygon
from PIL import Image
import os

# ===== I/O Helpers =====
def pick_and_record_keypoints(img1, img2, json_path):
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

    fig, (ax1, ax2) = plt.subplots(1, 2)
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

    plt.show()

    # Save the points as a JSON
    data = {
        "image1_keypoints": points_img1,
        "image2_keypoints": points_img2
    }
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return points_img1, points_img2

def display_img_with_keypoints(img, shape):
    plt.imshow(img, cmap='gray')
    y_coords, x_coords = zip(*shape)
    plt.scatter(x_coords, y_coords, color='red', marker='o', s=50)
    plt.show()

def display_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def display_triangulation(keypoints, triangulation):
    plt.figure()
    plt.triplot(keypoints[:, 1], keypoints[:, 0], triangulation.simplices, color='blue')
    plt.scatter(keypoints[:, 1], keypoints[:, 0], color='red')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()

# Saves a GIF, morphing forward then backward
def save_gif(morph_sequence, output_path, duration=100):
    images = [Image.fromarray((image_array).astype(np.uint8)) for image_array in morph_sequence]
    images_reversed = images[::-1]
    images[0].save(output_path, save_all=True, append_images=images + images_reversed, duration=duration, loop=0)

def parse_keypoints(path):
    keypoints = []
    with open(path, 'r') as file:
        lines = file.readlines()

        # Skip the first two lines (version and n_points) and the opening brace
        lines = lines[3:]

        # Parse each line containing coordinates, and stop at the closing brace
        for line in lines:
            if '}' in line:
                break
            x, y = map(float, line.split())
            keypoints.append([y, x])

    # Append the four corners to keypoints
    keypoints.extend([[0,0], [299,0], [299,249], [0,249]])
    return np.array(keypoints)


def plot_keypoints_and_vectors(base_keypoints, face_keypoints):
    # Compute the difference vectors
    difference_vectors = face_keypoints - base_keypoints
    
    # Separate the keypoints into x and y coordinates
    base_x, base_y = base_keypoints[:, 1], base_keypoints[:, 0]
    face_x, face_y = face_keypoints[:, 1], face_keypoints[:, 0] 
    diff_x, diff_y = difference_vectors[:, 1], difference_vectors[:, 0]

    plt.figure(figsize=(6, 6))
    
    # Plot base keypoints in one color
    plt.scatter(base_x, base_y, color='blue', label='Base Keypoints', s=50)
    
    # Plot face keypoints in another color
    plt.scatter(face_x, face_y, color='red', label='Face Keypoints', s=50)
    
    # Plot the difference vectors (arrows from base to face keypoints)
    for i in range(len(base_keypoints)):
        plt.arrow(base_x[i], base_y[i], diff_x[i], diff_y[i], color='green', 
                  head_width=2, length_includes_head=True, alpha=0.6)

    # Add labels and legend
    plt.legend()
    plt.gca().invert_yaxis()
    plt.title('Difference Vectors Between Base and Face Keypoints')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()



# ===== Algorithmic helpers =====

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

    if len(img.shape) == 2: # Grayscale image
        interpolator = RegularGridInterpolator((y, x), img, bounds_error=False, fill_value=None)
        interpolated_colors = interpolator(coords)
    else: # RGB image
        interpolated_colors = []
        for c in range(3):
            interpolator = RegularGridInterpolator((y, x), img[:, :, c], bounds_error=False, fill_value=None)
            interpolated_colors.append(interpolator(coords))
        interpolated_colors = np.stack(interpolated_colors, axis=-1)

    return interpolated_colors



# Morph an image into a new shape. Default triangulation is the Delaunay triangulation on the destination geometry.
def morph_into(src_name, src_im, src_pts, dst_name, dst_pts, tri=None, save_morphed=False):
    if tri is None:
        tri = Delaunay(dst_pts)

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

# Compute an intermediate morphing between 2 images
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
    intermediate_im = intermediate_im.astype(np.uint8)
    if save_intermediate:
        skio.imsave(f'../images/{im1_name}_{im2_name}[{warp_frac}:{dissolve_frac}]_intermediate.jpg', intermediate_im)
    return intermediate_im


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

        # Derive a triangulation for the key points using the average shape
        avg_pts = (im1_pts + im2_pts) / 2
        triangulation = Delaunay(avg_pts)
        
        display_triangulation(avg_pts, triangulation)
        display_triangulation(im1_pts, triangulation)
        display_triangulation(im2_pts, triangulation)

        for t in range(frames+1):
            morph_sequence.append(compute_intermediate(
                im1_name, im2_name, im1, im2, im1_pts, im2_pts, triangulation, warp_frac * t, dissolve_frac * t
            ))

        # Also save the midway face
        compute_intermediate(im1_name, im2_name, im1, im2, im1_pts, im2_pts, triangulation, 1/2, 1/2, save_intermediate=True)

    im_names_str = "_to_" .join(im_names)
    save_gif(morph_sequence, f'../images/{im_names_str}.gif')

# Computes the mean of all faces.
def compute_mean_face(faces, geometries):

    # Compute the average shape and triangulate
    stacked_geometries = np.stack(geometries, axis=0)
    average_geometry = np.mean(stacked_geometries, axis=0)
    triangulation = Delaunay(average_geometry)

    # Morph each image into the average shape
    mean_face = np.zeros_like(faces[0]).astype(np.float64)
    for i in range(len(faces)):
        morphed_face = morph_into('', faces[i], geometries[i], '', average_geometry, triangulation)
        mean_face += morphed_face.astype(np.float64)
        if i < 2:
            display_img(morphed_face.astype(np.uint8))

    # Compute the mean face by averaging all the morphed faces
    return (mean_face / len(faces)).astype(np.uint8)


# ===== Driver functions =====

# Gathers image and key points data, then runs morphing algorithm across all images
def morph(im_names, pick_keypoints=False):
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
        
        display_img_with_keypoints(im1, im1_keypoints)
        display_img_with_keypoints(im2, im2_keypoints)

        corresponding_keypoints.append((im1_keypoints, im2_keypoints))
    
    morph_multi(im_names, ims, corresponding_keypoints)


# Computes the mean face from the given data. Then morphs the given image into the mean face, and morphs the mean face into the given image.
def mean_face_driver(id, mean_face_name, im, im_name, new_mean_face=False, pick_keypoints=False):
    if new_mean_face:
        # Read in relevant data
        images_dir = '../data/brazilian_faces'
        keypoints_dir = '../data/brazilian_faces_keypoints'

        image_paths = sorted([f for f in os.listdir(images_dir) if id in f and f.endswith('.jpg')])
        keypoint_paths = sorted([f for f in os.listdir(keypoints_dir) if id in f and f.endswith('.pts')])

        full_image_paths = [images_dir + '/' + f for f in image_paths]
        full_keypoint_paths = [keypoints_dir + '/' + f for f  in keypoint_paths]

        print(full_image_paths[0], full_image_paths[1])

        faces = [skio.imread(path) for path in full_image_paths]
        geometries = [parse_keypoints(path) for path in full_keypoint_paths]

        # Compute mean face
        mean_face = compute_mean_face(faces, geometries)
        skio.imsave(f'../images/{mean_face_name}.jpg', mean_face)
    else:
        # Load mean face
        mean_face = skio.imread(f'../images/{mean_face_name}.jpg')

    # Define correspondances between image and mean face
    json_path = f'../data/{im_name}_{mean_face_name}.json'
    if pick_keypoints:
        im_keypoints, mean_face_keypoints = pick_and_record_keypoints(im, mean_face, json_path)
        im_keypoints, mean_face_keypoints = np.array(im_keypoints), np.array(mean_face_keypoints)
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)
        im_keypoints, mean_face_keypoints = np.array(data['image1_keypoints']), np.array(data['image2_keypoints'])

    # Morph image into the mean face, and vice versa
    morph_into(im_name, im, im_keypoints, mean_face_name, mean_face_keypoints, save_morphed=True)
    morph_into(mean_face_name, mean_face, mean_face_keypoints, im_name, im_keypoints, save_morphed=True)


# Produce an exaggerated version of a face relative to a base face.
def caricature(face_name, face, base_face_name, base_face, alpha=1.5, pick_keypoints=False):
    # Define correspondances
    json_path = f'../data/{face_name}_{base_face_name}.json'
    if pick_keypoints:
        face_keypoints, base_keypoints = pick_and_record_keypoints(face, base_face, json_path)
        face_keypoints, base_keypoints = np.array(face_keypoints), np.array(base_keypoints)
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)
        face_keypoints, base_keypoints = np.array(data['image1_keypoints']), np.array(data['image2_keypoints'])

    display_img_with_keypoints(face, face_keypoints)
    display_img_with_keypoints(base_face, base_keypoints)

    # Compute difference vectors for the key points
    difference_vectors = face_keypoints - base_keypoints
    plot_keypoints_and_vectors(base_keypoints, face_keypoints)

    # Compute target geometry (exaggerated face)
    target_keypoints = face_keypoints + alpha * difference_vectors

    # Clip the row and column coordinates to the image bounds
    target_keypoints[:, 0] = np.clip(target_keypoints[:, 0], 0, face.shape[0] - 1)
    target_keypoints[:, 1] = np.clip(target_keypoints[:, 1], 0, face.shape[1] - 1)

    # Morph face into exagerated geometry
    exaggerated_face = morph_into('', face, face_keypoints, '', target_keypoints)
    skio.imsave(f'../images/{face_name}_{base_face_name}_caracature[{alpha}].jpg', exaggerated_face)