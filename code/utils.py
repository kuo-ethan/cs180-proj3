import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import json
import skimage.io as skio

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
