import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "207826314"
ID2 = "208004119"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y
s_initial = np.array(s_initial).reshape((6, 1))  # make it a column vector

# Initialize particles by adding noise to initial state
# Make sure to create the right shape (6, N)
# Explicitly shape the noise scale to (6,1) for robust broadcasting
noise_std_initial = np.array([5, 5, 1, 1, 2, 2]).reshape(6, 1)
S = np.tile(s_initial, (1, N)) + np.random.normal(0, noise_std_initial, size=(6, N))


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Can handle s_prior in two formats:
    1. (6, N): Standard format where N is number of particles, each column is a particle.
    2. (1, 6*N): Flattened format, which will be internally reshaped to (6, N).

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift and noise, in (6, N) format.
    """
    
    num_particles = 0
    state_for_processing = None

    if s_prior.shape[0] == 6: # Standard (6, N) format
        state_for_processing = s_prior.copy() # Use a copy
        num_particles = s_prior.shape[1]
    elif s_prior.shape[0] == 1 and s_prior.shape[1] > 0 and s_prior.shape[1] % 6 == 0: # Flattened (1, 6*N) format
        num_particles = s_prior.shape[1] // 6
        # Reshape from (1, 6*N) to (6, N).
        # Original (1, 6N) is [x0,y0,w0,h0,vx0,vy0, x1,y1,w1,h1,vx1,vy1, ...]
        # Reshaping to (num_particles, 6) makes each row a particle's state.
        # Transposing then makes each column a particle's state.
        state_for_processing = s_prior.reshape(num_particles, 6).T
    else:
        raise ValueError(
            f"Unsupported s_prior shape: {s_prior.shape}. "
            f"Expected (6, N) or (1, 6*N where 6*N is total elements)."
        )

    state_for_processing = state_for_processing.astype(float)
    state_drifted = state_for_processing # Now guaranteed to be (6, num_particles)

    state_drifted[0, :] += state_drifted[4, :]  # x_c += vx
    state_drifted[1, :] += state_drifted[5, :]  # y_c += vy
    
    # Adding noise:
    # Add Gaussian noise to all components
    # std should be (6,1) for broadcasting with (6, num_particles)
    std = np.array([5, 5, 1, 1, 1, 1])[:, np.newaxis]
    noise = np.random.normal(0, std, size=(6, num_particles)) # Noise shape is (6, num_particles)
    state_drifted += noise
    
    # Keep state as float to preserve precision
    return state_drifted # Returns in (6, N) format
    



def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16, 16, 16))

    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Cropping patch of size (2*half width, 2*half height) centered at the specified location by state
    xc, yc, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
    
    # Handle boundary conditions
    x1 = max(0, int(xc - w))
    y1 = max(0, int(yc - h))
    x2 = min(img_width, int(xc + w))
    y2 = min(img_height, int(yc + h))
    
    # Check if the patch has valid dimensions
    if x2 <= x1 or y2 <= y1:
        return np.ones(16 * 16 * 16) / (16 * 16 * 16)  # Return uniform histogram if patch is invalid
    
    patch = image[y1:y2, x1:x2, :]  # shape: (2h, 2w, 3)

    # Quantize to 4-bit (values 0–15)
    quantized = patch // 16

    for pixel in quantized.reshape(-1, 3): # the reshape- to flatten quantized, each row now a pixel
        r, g, b = pixel
        hist[r, g, b] += 1

    hist = np.reshape(hist, 16 * 16 * 16)

    # normalize
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    else:
        hist = np.ones_like(hist) / hist.size  # Uniform distribution if empty patch
        
    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    N = previous_state.shape[1]
    S_next = np.zeros_like(previous_state)

    # Generate N random numbers between 0 and 1
    r = np.random.rand(N)

    # For each r, find the index j such that CDF[j] >= r[i]
    indices = np.searchsorted(cdf, r)

    # Use those indices to resample from previous_state
    S_next = previous_state[:, indices]

    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    # Ensure histograms are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate Bhattacharyya coefficient (similarity)
    bc = np.sum(np.sqrt(p * q))
    
    # Convert to distance (smaller value means more similar)
    # distance = -np.log(bc)
    
    # Since our implementation expects higher values for more similar histograms
    # Use an exponential function to map it back to a similarity measure
    similarity = np.exp(20 * bc)
    
    return similarity


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Ensure W is 1D array of shape (N,)
    if W.ndim != 1 or W.shape[0] != state.shape[1]:
        #This case should ideally not happen if W is prepared correctly
        print(f"Warning: W shape is {W.shape}, expected ({state.shape[1]},)")
        # Attempt to flatten W to (N,) and ensure it has the right number of elements
        W_corrected = W.flatten()
        if W_corrected.shape[0] != state.shape[1]:
             # Fallback to uniform weights if correction fails
            W_corrected = np.ones(state.shape[1]) / state.shape[1]
    else:
        W_corrected = W

    # Reshape W for broadcasting: (N,) -> (1, N)
    W_reshaped = W_corrected.reshape(1, -1)

    # Avg particle box: state is (6,N), W_reshaped is (1,N)
    # state * W_reshaped results in (6,N)
    # np.sum over axis=1 results in (6,)
    s_avg = np.sum(state * W_reshaped, axis=1)
    
    # Verify s_avg shape
    if s_avg.shape != (6,):
        raise ValueError(f"s_avg calculation error. Expected shape (6,), got {s_avg.shape}. State shape: {state.shape}, W_reshaped shape: {W_reshaped.shape}")

    x_avg, y_avg, w_avg, h_avg = s_avg[:4]
    x_avg -= w_avg #converting x,y to top left corner and not center
    y_avg -= h_avg
    w_avg *= 2 #converting w,h from halves to whole
    h_avg *= 2

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    i_max = np.argmax(W_corrected) # Use corrected W
    s_max = state[:, i_max]
    
    # Verify s_max shape
    if s_max.shape != (6,):
        raise ValueError(f"s_max calculation error. Expected shape (6,), got {s_max.shape}. Index i_max: {i_max}")
        
    x_max, y_max, w_max, h_max = s_max[:4]
    x_max -= w_max #converting x,y to top left corner and not center
    y_max -= h_max
    w_max *= 2 #converting w,h from halves to whole
    h_max *= 2

    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    # Ensure state_at_first_frame is (6,N) or (1, 6*N) as predict_particles can now handle both.
    # s_initial is (6,1).
    # Original request was to keep this .T version:
    # repmat(s_initial, N, 1) creates (6N, 1), .T makes it (1, 6N)
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    
    S = predict_particles(state_at_first_frame) # predict_particles will reshape this to (6,N)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = np.zeros(N)  # Change from (N, 1) to (N,) for consistent shape

    for i in range(N):
        p = compute_normalized_histogram(image, S[:,i:i+1])
        W[i] = bhattacharyya_distance(p, q)
    # No need to flatten since W is already 1D
    W /= np.sum(W) # normalize weights
    C = np.cumsum(W)

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    
    # Adaptive reference histogram (optional)
    q_adaptive = q.copy()
    alpha = 0.1  # Adaptation rate
    
    for image_name in image_name_list[1:]:
        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        W = np.zeros(N)  # Change from (N, 1) to (N,) for consistent shape
        for i in range(N):
            p = compute_normalized_histogram(current_image, S[:, i:i+1])
            W[i] = bhattacharyya_distance(p, q_adaptive)  # Use adaptive reference
        
        # Handle case where all weights could be zero
        if np.sum(W) > 0:
            W /= np.sum(W)
        else:
            # If all weights are zero, use uniform weights
            W = np.ones(N) / N
            
        C = np.cumsum(W)
        
        # Update reference histogram (adaptive tracking)
        if np.max(W) > 0.1:  # Only update if there's a confident match
            i_max = np.argmax(W)
            p_best = compute_normalized_histogram(current_image, S[:, i_max:i_max+1])
            q_adaptive = (1 - alpha) * q_adaptive + alpha * p_best

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed % 10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
