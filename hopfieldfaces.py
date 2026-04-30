import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from skimage.filters import threshold_otsu

def train_hopfield(patterns):
    """Hebbian learning rule to calculate the weight matrix."""
    num_patterns, num_neurons = patterns.shape
    W = np.zeros((num_neurons, num_neurons))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W / num_neurons

def recall_async(W, pattern, epochs=3):
    """Asynchronous recall for guaranteed energy descent."""
    current_state = np.copy(pattern)
    num_neurons = len(pattern)
    for _ in range(epochs):
        update_order = np.random.permutation(num_neurons)
        for i in update_order:
            net_input = np.dot(W[i], current_state)
            current_state[i] = 1 if net_input > 0 else -1
    return current_state

# Number of pixels that differ in the same position
def hamming_distance(target, recall):
    return np.sum(target != recall)

# Percentage of correctly recalled pixels
def pixel_accuracy(target, recall):
    return (1 - (np.sum(target != recall) / len(target))) * 100

def memory_capacity_experiment():
    loading_conditions = [2, 6, 10, 15, 40, 100]

    fig, axes = plt.subplots(4, 8, figsize=(22, 11))

    for row, target_idx in enumerate(target_indices):
        print(f"\nProcessing Subject {row + 1}...")
        target_face = binary_patterns[target_idx]
        
        # Create the Corrupted Cue (50% missing data / forced to black)
        corrupted_cue = np.copy(target_face)
        noise_indices = np.random.choice(N, size=int(0.5 * N), replace=False)
        corrupted_cue[noise_indices] = -1

        # Calculate metrics
        distance = hamming_distance(target_face, corrupted_cue)
        accuracy = pixel_accuracy(target_face, corrupted_cue)
        
        # Plot Original and Cue
        axes[row, 0].imshow(target_face.reshape(64, 64), cmap='Greys_r')
        axes[row, 1].imshow(corrupted_cue.reshape(64, 64), cmap='Greys_r')
        axes[row, 1].set_xlabel(f'Hamming Distance: {distance}\nPixel Accuracy: {int(accuracy)}%', fontsize=12)
        
        if row == 0:
            axes[row, 0].set_title('Original Target', fontweight='bold', fontsize=12)
            axes[row, 1].set_title('Cue (50% Missing)', fontweight='bold', fontsize=12)
            
        for col, count in enumerate(loading_conditions):
            # Build a training set specifically for this condition
            other_faces = np.delete(binary_patterns, target_idx, axis=0)
            np.random.shuffle(other_faces) # Randomize the background faces each time
            training_set = np.vstack([target_face, other_faces[:count - 1]])
            
            # Train and Recall
            W = train_hopfield(training_set)
            recall = recall_async(W, corrupted_cue, epochs=3)

            # Calculate metrics
            distance = hamming_distance(target_face, recall)
            accuracy = pixel_accuracy(target_face, recall)
            
            # Plot Results
            axes[row, col + 2].imshow(recall.reshape(64, 64), cmap='Greys_r')
            axes[row, col + 2].set_xlabel(f'Hamming Distance: {distance}\nPixel Accuracy: {int(accuracy)}%', fontsize=12)
            if row == 0:
                axes[row, col + 2].set_title(f'Recall ({count} Faces)', fontweight='bold', fontsize=12)

    # Format and Display
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Complete Memory Degradation Spectrum (Random Subjects)', y=0.98, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

# --- 1. Fetch and Preprocess the Faces ---
print("Downloading and processing Olivetti Faces...")
faces_data = fetch_olivetti_faces(shuffle=False) 
images = faces_data.images 

binary_patterns = []
for img in images:
    thresh = threshold_otsu(img)
    binary_patterns.append(np.where(img > thresh, 1, -1).flatten())
binary_patterns = np.array(binary_patterns)
N = binary_patterns.shape[1] 

# Select 4 random distinct people (out of 40)
random_people = np.random.choice(40, size=4, replace=False)
# Grab the first photo index for each of those 4 people
target_indices = random_people * 10 

print(f"\nRunning experiments on randomly selected people (IDs: {random_people})...")

# Run experiments
memory_capacity_experiment()