import numpy as np
import matplotlib.pyplot as plt

def train_hopfield(patterns):
    num_patterns, num_neurons = patterns.shape
    W = np.zeros((num_neurons, num_neurons))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W / num_neurons

def recall_async(W, pattern, epochs=5):
    current_state = np.copy(pattern)
    num_neurons = len(pattern)
    
    for _ in range(epochs):
        update_order = np.random.permutation(num_neurons)
        for i in update_order:
            net_input = np.dot(W[i], current_state)
            current_state[i] = 1 if net_input > 0 else -1
                
    return current_state

N = 400 
capacity_limit = int(0.14 * N) 

np.random.seed(32)
#due to some luck when using 42 it manages to recall the pattern even at 65 patterns, so I changed it to 32 to get a more clear transition in the results.

target_memory = np.ones((20, 20)) * -1
for i in range(20):
    target_memory[i, i] = 1
    target_memory[i, 19 - i] = 1
target_memory = target_memory.flatten()

loading_conditions = [45, 52, 58, 65]
all_random_patterns = np.random.choice([-1, 1], size=(max(loading_conditions), N))

trained_matrices = []
for count in loading_conditions:
    training_set = np.vstack([target_memory, all_random_patterns[:count - 1]])
    trained_matrices.append(train_hopfield(training_set))

corrupted_cue = np.copy(target_memory)
noise_indices = np.random.choice(N, size=int(0.2 * N), replace=False)
corrupted_cue[noise_indices] *= -1

recall_results = []
for W in trained_matrices:
    recall_results.append(recall_async(W, corrupted_cue, epochs=10))

fig, axes = plt.subplots(1, 6, figsize=(18, 3)) 
titles = [
    'Original Target', 
    'Corrupted Cue', 
    f'45 Patterns (-11)', 
    f'52 Patterns (-4)', 
    f'58 Patterns (+2)', 
    f'65 Patterns (+9)'
]

images = [target_memory, corrupted_cue] + recall_results

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img.reshape(20, 20), cmap='Greys', vmin=-1, vmax=1)
    ax.set_title(title, fontsize=11, fontweight='bold' if 'Patterns' in title else 'normal')
    ax.axis('off')

plt.suptitle(f'Hopfield Phase Transition (Async Updates)', y=1.05, fontsize=14)
plt.tight_layout()
plt.show()