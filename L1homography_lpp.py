import numpy
import cv2
import matplotlib.pyplot as plt
import argparse
from os.path import join
import pulp as lpp
import numpy as np
import scipy.linalg as scm
import time

def get_homography_matrix(video, transformation_model='homography'):
    """
    Computes the homography matrix for consecutive frames in a video.
    
    Args:
        video: VideoCapture object opened with cv2.VideoCapture.
        transformation_model: The model to use for frame transformation. 
                              Options are 'rigid', 'affine', and 'homography'.
                              
    Returns:
        A numpy array containing the homography matrices for all frame transitions.
    """
    
    # Set video to the first frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Read the first frame
    success, frame = video.read()
    if not success:
        print("Error in reading frame")
    else:                 
        # Convert the first frame to grayscale
        prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize a 3D array to store homography matrices
    homographies = np.zeros((total_frames - 1, 3, 3), np.float32)
    
    # Counter for the number of valid frames processed
    processed_frames = 1

    while processed_frames < total_frames-1:
    
        # Detect features in the previous frame
        src_points = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=3)
        
        # Read the next frame
        success, frame = video.read()

        if not success:
            print("Error in reading frame no:", processed_frames)
            break
            
        if frame is None:
            print("Empty frame detected")
            break
            
        # Convert the current frame to grayscale
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow to track feature points
        dst_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, src_points, None)
        
        # Ensure the shape of source and destination points are the same
        assert src_points.shape == dst_points.shape
        
        # Filter points based on the status vector
        valid_idx = np.where(status == 1)[0]
        
        src_points = src_points[valid_idx]
        dst_points = dst_points[valid_idx]
        
        # Compute transformation matrix based on the specified model
        if transformation_model == 'rigid':
            matrix, mask = cv2.estimateAffinePartial2D(dst_points, src_points)
            homographies[processed_frames-1, 0:2, 0:3] = matrix
            homographies[processed_frames-1, 2, 2] = 1
        elif transformation_model == 'affine':
            matrix, mask = cv2.estimateAffine2D(dst_points, src_points)
            homographies[processed_frames-1, 0:2, 0:3] = matrix
            homographies[processed_frames-1, 2, 2] = 1
        elif transformation_model == 'homography':
            matrix, mask = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 1.0)
            det_value = np.sqrt(np.linalg.det(matrix))
            homographies[processed_frames-1, :, :] = np.divide(matrix, det_value)
        else:
            print("Invalid model")
            
        # Prepare for next iteration
        prev_frame_gray = curr_frame_gray
        processed_frames += 1

        
    # Adjust the size of the homography array to match the number of valid frames processed
    homographies = homographies[:processed_frames-1, :, :]
    
    # Prepend an identity matrix for the first frame
    homographies = np.concatenate((np.reshape(np.eye(3), (1, 3, 3)), homographies), axis=0)
    
    return homographies
    
def get_crop_window_coordinates(image_dimensions, crop_ratio):
    """
    Calculate the corner points for a crop window based on a given crop ratio.
    
    The function computes a cropping area that maintains the aspect ratio of the original image.
    It centers the crop area in the middle of the image.
    
    Parameters:
    - image_dimensions: A tuple or list with two elements (width, height) representing the dimensions of the original image.
    - crop_ratio: A float representing the ratio of the crop dimensions relative to the original image dimensions.
    
    Returns:
    - A list of tuples, where each tuple represents the (x, y) coordinates of each corner point of the crop window.
    """
    
    # Calculate the center of the image
    center_x = image_dimensions[0] / 2
    center_y = image_dimensions[1] / 2
    
    # Determine dimensions of the crop window
    crop_width = image_dimensions[0] * crop_ratio
    crop_height = image_dimensions[1] * crop_ratio
    
    # Calculate top-left corner of the crop window to maintain it centered
    top_left_x = -crop_width / 2
    top_left_y = -crop_height / 2
    
    # Calculate the coordinates of the crop window's corners
    corner_points = [
        (top_left_x, top_left_y),  # Top-left corner
        (top_left_x + crop_width, top_left_y),  # Top-right corner
        (top_left_x + crop_width, top_left_y + crop_height),  # Bottom-right corner
        (top_left_x, top_left_y + crop_height)  # Bottom-left corner
    ]
    
    return corner_points
    
def stabilize_video(Fmat, frame_shape, no_frames, crop_ratio, w0, w1, w2, w3):
    """
    Stabilizes a sequence of homography matrices for video frame stabilization.

    Parameters:
    - Fmat: A sequence of homography matrices.
    - frame_shape: The shape of the video frames (width, height).
    - no_frames: The number of frames in the video sequence.
    - crop_ratio: The ratio for cropping the frames to stabilize the video.
    - w0: Weight for the stabilization term.
    - w1: Weight for the first smoothness term.
    - w2: Weight for the second smoothness term.
    - w3: Weight for the third smoothness term.

    Returns:
    - A sequence of stabilized homography matrices.
    """
    # Define the frame dimensions and the number of transformation parameters.
    [w, h] = frame_shape
    N = 9
    
    # Initialize a list to store the logarithm of the transformation matrices.
    fmat = []
    for frame_id in range(no_frames):
        # Compute the matrix logarithm of each transformation matrix.
        tempmat = scm.logm(Fmat[frame_id, :, :])
        # Flatten and store the log-transformed matrix.
        fmat.append(np.reshape(tempmat, (N, )))
        
    # Define a vector of ones for weighting in the optimization problem.
    c1 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    # Calculate the coordinates of the corners after applying the crop ratio.
    corner_points = get_crop_window_coordinates(frame_shape, crop_ratio)
    
    # Initialize dictionaries to store coefficients for X and Y transformations.
    Ax = {}
    Ay = {}
    
    # Populate the Ax and Ay with transformation coefficients for corner points.
    for ind in range(len(corner_points)):
        # Coefficients for X transformations.
        Ax[ind] = np.array([corner_points[ind][0], corner_points[ind][1], 1., 0., 0., 0., -corner_points[ind][0]**2, -corner_points[ind][0]*-corner_points[ind][1], -corner_points[ind][0]])
        # Coefficients for Y transformations.
        Ay[ind] = np.array([0., 0., 0., corner_points[ind][0], corner_points[ind][1], 1., -corner_points[ind][0]*-corner_points[ind][1], -corner_points[ind][1]**2, -corner_points[ind][1]])

    # Define the threshold for the minimum allowable area after cropping.
    threshold_area = (1 - crop_ratio)**2

    # Calculate the initial area of the crop window.
    Area = (corner_points[1][0] - corner_points[0][0]) * (corner_points[2][1] - corner_points[1][1])
    Area -= (corner_points[2][0] - corner_points[1][0]) * (corner_points[1][1] - corner_points[0][1])
    Area += (corner_points[3][0] - corner_points[3][0]) * (corner_points[0][1] - corner_points[3][1])
    Area -= (corner_points[0][0] - corner_points[3][0]) * (corner_points[3][1] - corner_points[3][1])
    Area = Area/2

    # Calculating gradients of the area with respect to corner points.
    Area_grad = []
    Area_grad.append([0.5*(corner_points[1][1] - corner_points[3][1]), 0.5*(corner_points[3][0] - corner_points[1][0])])
    Area_grad.append([0.5*(corner_points[2][1] - corner_points[0][1]), 0.5*(corner_points[0][0] - corner_points[2][0])])
    Area_grad.append([0.5*(corner_points[3][1] - corner_points[1][1]), 0.5*(corner_points[1][0] - corner_points[3][0])])
    Area_grad.append([0.5*(corner_points[0][1] - corner_points[2][1]), 0.5*(corner_points[2][0] - corner_points[0][0])])

    # Define the threshold for the minimum allowable length after cropping.
    threshold_length = 1. - crop_ratio

    # Initialize and compute the gradients of lengths for the four sides of the crop window.
    length1 = 0.5 * (w**2)
    length1_grad = []
    length1_grad.append([corner_points[0][0] - corner_points[1][0], corner_points[0][1] - corner_points[1][1]])
    length1_grad.append([corner_points[1][0] - corner_points[0][0], corner_points[1][1] - corner_points[0][1]])
    length1_grad.append([0.0, 0.0])
    length1_grad.append([0.0, 0.0])

    length2 = 0.5 * (h**2)
    length2_grad = []
    length2_grad.append([0.0, 0.0])
    length2_grad.append([corner_points[1][0] - corner_points[2][0], corner_points[1][1] - corner_points[2][1]])
    length2_grad.append([corner_points[2][0] - corner_points[1][0], corner_points[2][1] - corner_points[1][1]])
    length2_grad.append([0.0, 0.0])    

    length3 = 0.5 * (w**2)
    length3_grad = []
    length3_grad.append([0.0, 0.0])
    length3_grad.append([0.0, 0.0])    
    length3_grad.append([corner_points[2][0] - corner_points[3][0], corner_points[2][1] - corner_points[3][1]])
    length3_grad.append([corner_points[3][0] - corner_points[2][0], corner_points[3][1] - corner_points[2][1]])

    length4 = 0.5 * (h**2)
    length4_grad = []
    length4_grad.append([corner_points[0][0] - corner_points[3][0], corner_points[0][1] - corner_points[3][1]])
    length4_grad.append([0.0, 0.0])
    length4_grad.append([0.0, 0.0])    
    length4_grad.append([corner_points[3][0] - corner_points[0][0], corner_points[3][1] - corner_points[0][1]])

    # Initialize the optimization problem with the goal of minimizing the objective function.
    prob = lpp.LpProblem("stabilize", lpp.LpMinimize)
    n_frames = len(Fmat)

    # Stabilization parameters for error in log homography space
    p = lpp.LpVariable.dicts("p", ((i,j) for i in range(n_frames) for j in range(N)))
    # Parameters to maintain first order smoothness 
    e1 = lpp.LpVariable.dicts("e1", ((i,j) for i in range(n_frames-1) for j in range(N)))
    # Parameters to maintain second order smoothness 
    e2 = lpp.LpVariable.dicts("e2", ((i,j) for i in range(n_frames-2) for j in range(N)))
    # Parameters to maintain third order smoothness 
    e3 = lpp.LpVariable.dicts("e3", ((i,j) for i in range(n_frames-3) for j in range(N)))

    # Corresponding slack variables
    tp = lpp.LpVariable.dicts("tp", ((i,j) for i in range(n_frames) for j in range(N)), lowBound=0.)
    te1 = lpp.LpVariable.dicts("te1", ((i,j) for i in range(n_frames-1) for j in range(N)), lowBound=0.)
    te2 = lpp.LpVariable.dicts("te2", ((i,j) for i in range(n_frames-2) for j in range(N)), lowBound=0.)
    te3 = lpp.LpVariable.dicts("te3", ((i,j) for i in range(n_frames-3) for j in range(N)), lowBound=0.)

    # Objective function
    prob += w0 * lpp.lpSum([tp[frame_id, j] * c1[j] for frame_id in range(n_frames) for j in range(N)]) + \
            w1 * lpp.lpSum([te1[frame_id, j] * c1[j] for frame_id in range(n_frames-1) for j in range(N)]) + \
            w2 * lpp.lpSum([te2[frame_id, j] * c1[j] for frame_id in range(n_frames-2) for j in range(N)]) + \
            w3 * lpp.lpSum([te3[frame_id, j] * c1[j] for frame_id in range(n_frames-3) for j in range(N)]) 

    # Set constraints for the optimization problem to ensure the transformations are valid and within bounds.   
    for frame_id in range(n_frames):

        # Trace constraints
        prob += p[frame_id, 0] + p[frame_id, 4] + p[frame_id, 7] <= 0.
        prob += p[frame_id, 0] + p[frame_id, 4] + p[frame_id, 7] >= 0.

        # Bound constraints
        prob += -1.1 <= p[frame_id, 0]   
        prob += p[frame_id, 0] <= 1.1
        prob += -1.1 <= p[frame_id, 4]   
        prob += p[frame_id, 4] <= 1.1
        prob += -1.1 <= p[frame_id, 8]   
        prob += p[frame_id, 8] <= 1.1
        prob += -0.1 <= p[frame_id, 1]   
        prob += p[frame_id, 1] <= 0.1 
        prob += -0.1 <= p[frame_id, 3]   
        prob += p[frame_id, 3] <= 0.1     
        prob += -100 <= p[frame_id, 2]   
        prob += p[frame_id, 2] <= 100
        prob += -100 <= p[frame_id, 5]   
        prob += p[frame_id, 5] <= 100   
        prob += -0.00001 <= p[frame_id, 6]   
        prob += p[frame_id, 6] <= 0.00001 
        prob += -0.00001 <= p[frame_id, 7]   
        prob += p[frame_id, 7] <= 0.00001 

        # Distortion constraints
        prob += p[frame_id, 0] - p[frame_id, 4] <= 0.1
        prob += -0.1 <= p[frame_id, 0] - p[frame_id, 4]         
        prob += p[frame_id, 1] + p[frame_id, 3] <= 0.05
        prob += -0.05 <= p[frame_id, 1] + p[frame_id, 3]

    # L1 norm constraints - required to convert to linear program
    for frame_id in range(n_frames):
        for j in range(N):
            prob += p[frame_id, j] <= tp[frame_id, j]
            prob += -p[frame_id,j] <= tp[frame_id, j]

    for frame_id in range(n_frames-1):
        for j in range(N):
            prob += e1[frame_id, j] <= te1[frame_id, j]
            prob += -e1[frame_id,j] <= te1[frame_id, j]  

    for frame_id in range(n_frames-2):
        for j in range(N):
            prob += e2[frame_id, j] <= te2[frame_id, j]
            prob += -e2[frame_id,j] <= te2[frame_id, j]  

    for frame_id in range(n_frames-3):
        for j in range(N):
            prob += e3[frame_id, j] <= te3[frame_id, j]
            prob += -e3[frame_id,j] <= te3[frame_id, j]  

    # Constraints corresponding to slack variables representing derivatives
    for frame_id in range(n_frames-1):
        for j in range(N):
            prob += e1[frame_id, j] - p[frame_id+1, j] + p[frame_id, j] <= fmat[frame_id+1][j]
            prob += e1[frame_id, j] - p[frame_id+1, j] + p[frame_id, j] >= fmat[frame_id+1][j]

    for frame_id in range(n_frames-2):
        for j in range(N):
            prob += e2[frame_id, j] - p[frame_id+2, j] + 2*p[frame_id+1, j] - p[frame_id, j] <= fmat[frame_id+2][j] - fmat[frame_id+1][j]
            prob += e2[frame_id, j] - p[frame_id+2, j] + 2*p[frame_id+1, j] - p[frame_id, j] >= fmat[frame_id+2][j] - fmat[frame_id+1][j]

    for frame_id in range(n_frames-3):
        for j in range(N):
            prob += e3[frame_id, j] - p[frame_id+3, j] + 3*p[frame_id+2, j] - 3*p[frame_id+1, j] + p[frame_id, j] <= fmat[frame_id+3][j] -2*fmat[frame_id+2][j] + fmat[frame_id+1][j]
            prob += e3[frame_id, j] - p[frame_id+3, j] + 3*p[frame_id+2, j] - 3*p[frame_id+1, j] + p[frame_id, j] >= fmat[frame_id+3][j] -2*fmat[frame_id+2][j] + fmat[frame_id+1][j]


    # Calculate the change in position for each corner point after applying transformations.
    changeposx = [[0.] * 4] * (no_frames)
    changeposy = [[0.] * 4] * (no_frames)

    for frame_id in range(no_frames):
        for j in range(len(corner_points)):
            changeposx[frame_id][j] = p[frame_id, 0] * Ax[j][0] + p[frame_id, 1] * Ax[j][1] + p[frame_id, 2] * Ax[j][2]
            + p[frame_id, 3] * Ax[j][3] + p[frame_id, 4] * Ax[j][4] + p[frame_id, 5] * Ax[j][5]
            + p[frame_id, 6] * Ax[j][6] + p[frame_id, 7] * Ax[j][7] + p[frame_id, 8] * Ax[j][8]

            changeposy[frame_id][j] = p[frame_id, 0] * Ay[j][0] + p[frame_id, 1] * Ay[j][1] + p[frame_id, 2] * Ay[j][2]
            + p[frame_id, 3] * Ay[j][3] + p[frame_id, 4] * Ay[j][4] + p[frame_id, 5] * Ay[j][5]
            + p[frame_id, 6] * Ay[j][6] + p[frame_id, 7] * Ay[j][7] + p[frame_id, 8] * Ay[j][8]

   # Enforce constraints on the transformed positions to ensure they remain within the frame boundaries.
    for frame_id in range(no_frames):
        for j in range(len(corner_points)):
            prob += changeposx[frame_id][j] + corner_points[j][0] >= -w/2
            prob += changeposx[frame_id][j] + corner_points[j][0] <= w/2
            prob += changeposy[frame_id][j] + corner_points[j][1] >= -h/2
            prob += changeposy[frame_id][j] + corner_points[j][1] <= h/2

    Area_tot = [None] * (no_frames)
    for frame_id in range(no_frames):
        Area_tot[frame_id] = Area
        for j in range(len(corner_points)):
            Area_tot[frame_id] += Area_grad[j][0] * changeposx[frame_id][j]
            Area_tot[frame_id] += Area_grad[j][1] * changeposy[frame_id][j]
        prob += Area_tot[frame_id] >= threshold_area * w * h 


    length1_tot = [None] * (no_frames)
    for frame_id in range(no_frames):
        length1_tot[frame_id] = length1
        for j in range(len(corner_points)):
            length1_tot[frame_id] += length1_grad[j][0] * changeposx[frame_id][j]
            length1_tot[frame_id] += length1_grad[j][1] * changeposy[frame_id][j]
        prob += length1_tot[frame_id] >= threshold_length * length1

    length2_tot = [None] * (no_frames)
    for frame_id in range(no_frames):
        length2_tot[frame_id] = length2
        for j in range(len(corner_points)):
            length2_tot[frame_id] += length2_grad[j][0] * changeposx[frame_id][j]
            length2_tot[frame_id] += length2_grad[j][1] * changeposy[frame_id][j]
        prob += length2_tot[frame_id] >= threshold_length * length2

    length3_tot = [None] * (no_frames)
    for frame_id in range(no_frames):
        length3_tot[frame_id] = length3
        for j in range(len(corner_points)):
            length3_tot[frame_id] += length3_grad[j][0] * changeposx[frame_id][j]
            length3_tot[frame_id] += length3_grad[j][1] * changeposy[frame_id][j]
        prob += length3_tot[frame_id] >= threshold_length * length3

    length4_tot = [None] * (no_frames)
    for frame_id in range(no_frames):
        length4_tot[frame_id] = length4
        for j in range(len(corner_points)):
            length4_tot[frame_id] += length4_grad[j][0] * changeposx[frame_id][j]
            length4_tot[frame_id] += length4_grad[j][1] * changeposy[frame_id][j]
        prob += length4_tot[frame_id] >= threshold_length * length4

    # Solve the optimization problem.
    prob.solve()

    # Construct the output transformation matrices from the optimized variables.
    out = np.zeros((no_frames, 3, 3))

    for frame_id in range(no_frames):
        outmat = np.array([[p[frame_id, 0].varValue, p[frame_id, 1].varValue, p[frame_id, 2].varValue],
                           [p[frame_id, 3].varValue, p[frame_id, 4].varValue, p[frame_id, 5].varValue], 
                           [p[frame_id, 6].varValue, p[frame_id, 7].varValue, p[frame_id, 8].varValue]])
        
        out[frame_id, :, :] = scm.expm(outmat)

    return out

def stabilize_and_crop_video(input_cap, output_file, frame_rate, transform_matrices, input_frame_shape, crop_factor):
    """
    Stabilizes and crops a video based on provided transformation matrices and cropping factor.
    
    Parameters:
    - input_cap: cv2.VideoCapture object, the video capture object for the input video.
    - output_file: str, the path to the output video file.
    - frame_rate: float, the frame rate for the output video.
    - transform_matrices: numpy.ndarray, the transformation matrices for each frame stabilization.
    - input_frame_shape: list or tuple, the shape of frames in the input video (height, width).
    - crop_factor: float, the factor to crop the video by (0 to 1, where 1 means no cropping).
    
    The function saves the stabilized and cropped video to the specified output file.
    """
    
    # Calculate the number of frames based on the provided transformation matrices
    num_frames = len(transform_matrices)

    # Set the video capture to the first frame
    input_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Calculate the cropping values based on the input frame shape and crop factor
    crop_values = input_frame_shape.copy()
    crop_values[0] = int(input_frame_shape[0] * (1 - crop_factor) / 2)
    crop_values[1] = int(input_frame_shape[1] * (1 - crop_factor) / 2)

    # Calculate the output frame shape after cropping
    output_frame_shape = input_frame_shape.copy()
    output_frame_shape[0] = input_frame_shape[0] - 2*crop_values[0]
    output_frame_shape[1] = input_frame_shape[1] - 2*crop_values[1]

    # Initialize the VideoWriter object for the output video
    output_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, output_frame_shape)

    # Initialize a counter for valid frames processed
    num_valid_frames = 0

    # Process each frame
    while num_valid_frames < num_frames:
        # Read the next frame
        success, frame = input_cap.read()

        if not success or frame is None:
            print("Error in reading frame")
            break

        # Retrieve the corresponding transformation matrix and invert it for stabilization
        B_matrix = transform_matrices[num_valid_frames, :, :]
        transformation_matrix = np.linalg.inv(B_matrix)

        # Apply the inverse transformation to stabilize the frame
        frame_stabilized = cv2.warpPerspective(frame, transformation_matrix, input_frame_shape, flags=cv2.INTER_CUBIC)

        # Crop the stabilized frame according to the calculated crop values
        frame_cropped = frame_stabilized[crop_values[1]: crop_values[1] + output_frame_shape[1], crop_values[0]: crop_values[0] + output_frame_shape[0], :]

        # Write the cropped and stabilized frame to the output file
        output_writer.write(frame_cropped)

        # Increment the valid frames counter
        num_valid_frames += 1

    # Release the VideoWriter object to finalize the output video file
    output_writer.release()


def plot_trajectory_comparison(input_trajectory, stabilized_trajectory, trajectory_path):
    """
    Plots the comparison between input and stabilized trajectories for video frames.
    
    Parameters:
    - input_trajectory: A numpy array with shape (N, 3) for the input video, where N is the number of frames.
      Each row represents [x, y, z] coordinates for a frame.
    - stabilized_trajectory: A similar array for the stabilized video.
    
    The function plots the x and y coordinates (normalized by the z coordinate) across all frames,
    illustrating the difference between the input and stabilized trajectories.
    """
    
    # Plot X coordinate trajectory comparison
    plt.figure()
    # Normalize x coordinate by z to project to 2D and plot for input trajectory
    plt.plot(np.divide(input_trajectory[:, 0], input_trajectory[:, 2]), label='Input')
    # Repeat for stabilized trajectory
    plt.plot(np.divide(stabilized_trajectory[:, 0], stabilized_trajectory[:, 2]), label='Stabilized')
    plt.xlabel('Frame number')
    plt.ylabel('X coordinate (pixels)')
    plt.title('Input vs Stabilized X Trajectory')
    plt.legend()
    plt.savefig(trajectory_path + "_X_trajectory_comparison.png")
    plt.close()
    
    # Plot Y coordinate trajectory comparison
    plt.figure()
    # Normalize y coordinate by z to project to 2D and plot for input trajectory
    plt.plot(np.divide(input_trajectory[:, 1], input_trajectory[:, 2]), label='Input')
    # Repeat for stabilized trajectory
    plt.plot(np.divide(stabilized_trajectory[:, 1], stabilized_trajectory[:, 2]), label='Stabilized')
    plt.xlabel('Frame number')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Input vs Stabilized Y Trajectory')
    plt.legend()
    plt.savefig(trajectory_path + "_Y_trajectory_comparison.png")
    plt.close()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Video Stabilization Script")

    # Define arguments for input file, output file, and crop ratio
    parser.add_argument("-i", action="store", dest="file")
    parser.add_argument("-o", action="store", dest="file_out")
    parser.add_argument("-crop_ratio", action="store", dest="crop_ratio", type=float)

    # Parse command line arguments
    args_read = parser.parse_args()

    in_file = args_read.file
    out_file = args_read.file_out
    crop_ratio = args_read.crop_ratio

    # Extract filename and video type from output file path
    out_file_folder = ''.join(out_file.split('.')[:-1])
    [filename, vid_type] = out_file.split('/')[-1].split('.')

    # Determine video codec based on file type
    if vid_type == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    elif vid_type == "avi":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        print("Unsupported video format")
        exit(-1)

    # Open input video file
    video = cv2.VideoCapture(in_file)

    # Get video properties
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    print("Finding Homography")
    fhomo_start = time.time()
    Hmat = get_homography_matrix(video, 'homography')
    fhomo_end = time.time()

    print("Stabilizing frames")
    fstabilize_start = time.time()
    no_frames = len(Hmat)

    # Initialize weights for stabilization
    w0 = 1
    w1 = 1.
    w2 = 10000.
    w3 = 100.

    Bmat = stabilize_video(Hmat, [w, h], no_frames, crop_ratio, w0, w1, w2, w3)
    fstabilize_end = time.time()

    print("Warping and write output")

    fwarp_start = time.time()
    stabilize_and_crop_video(video, out_file, fps, Bmat, [w, h], crop_ratio)
    fwarp_end = time.time()

    # Print timings for each stage
    time_homo = fhomo_end - fhomo_start
    time_stab = fstabilize_end - fstabilize_start
    time_warp = fwarp_end - fwarp_start
    total_time = time_homo + time_stab + time_warp
    print(f"Time taken for finding homography: {time_homo} seconds")
    print(f"Time taken for stabilization: {time_stab} seconds")
    print(f"Time taken for warping: {time_warp} seconds")
    print(f"Total Time taken: {total_time} seconds")

    print("Plotting Trajectories")

    # Calculate input and output trajectories for plotting
    F_t = Hmat.copy()
    for i in range(no_frames):
        F_t[i, :, :] = F_t[i-1, :, :] @ Hmat[i, :, :]

    S_t = F_t.copy()
    for i in range(no_frames):
        S_t[i, :, :] = S_t[i, :, :] @ Bmat[i, :, :]

    start_point = np.array([0, 0, 1]).T

    trajectory_input = F_t @ start_point
    trajectory_output = S_t @ start_point
    plot_trajectory_comparison(trajectory_input, trajectory_output, out_file_folder)




