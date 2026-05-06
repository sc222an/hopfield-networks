import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from skimage.filters import threshold_otsu
from mpl_toolkits.mplot3d import Axes3D

# Network functions

def train_hopfield(patterns):
    """Hebbian learning rule to calculate the weight matrix."""
    num_patterns, num_neurons = patterns.shape
    W = np.zeros((num_neurons, num_neurons))

    print(f"{num_patterns} patterns with {num_neurons} neurons each.")

    for p in patterns:
        W += np.outer(p, p)

    np.fill_diagonal(W, 0)
    return W / num_neurons


def hopfield_energy(W, state):
    """Calculate Hopfield energy: E = -0.5 * s^T W s."""
    return -0.5 * state @ W @ state


def recall_async(W, pattern, epochs=3, track_energy=True):
    """
    Asynchronous recall.
    Updates neurons one-by-one in random order.
    Returns final state and energy history.
    """
    current_state = np.copy(pattern)
    num_neurons = len(pattern)

    energy_history = []

    if track_energy:
        energy_history.append(hopfield_energy(W, current_state))

    for epoch in range(epochs):
        previous_state = np.copy(current_state)

        update_order = np.random.permutation(num_neurons)

        for i in update_order:
            net_input = np.dot(W[i], current_state)
            current_state[i] = 1 if net_input > 0 else -1

        if track_energy:
            energy_history.append(hopfield_energy(W, current_state))

        # Stop early if the state no longer changes
        if np.array_equal(previous_state, current_state):
            break

    epochs_used = epoch + 1

    return current_state, energy_history, epochs_used

# Eval functions

def pixel_accuracy(original, recalled):
    """Percentage of pixels that match the original pattern."""
    return np.mean(original == recalled) * 100


def hamming_distance(original, recalled):
    """Number of pixels that differ from the original pattern."""
    return np.sum(original != recalled)


def pattern_load_ratio(num_patterns, num_neurons):
    """Pattern load ratio alpha = P / N."""
    return num_patterns / num_neurons


def capacity_percentage(num_patterns, num_neurons):
    """Percentage of the theoretical Hopfield capacity used."""
    estimated_capacity = 0.14 * num_neurons
    return (num_patterns / estimated_capacity) * 100

# Energy plotting functions

def plot_energy_landscape(W, pattern_a, pattern_b, title="Projected Hopfield Energy Landscape"):
    """
    Visualise a 2D projection of the energy landscape.

    The real Hopfield landscape is 4096-dimensional.
    This function projects it onto two directions:
    - pattern_a, usually the target face
    - pattern_b, usually another stored face

    Each point on the grid is converted back into a bipolar state using sign().
    """
    x_vals = np.linspace(-1.5, 1.5, 60)
    y_vals = np.linspace(-1.5, 1.5, 60)

    energy_grid = np.zeros((len(y_vals), len(x_vals)))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            combined_state = x * pattern_a + y * pattern_b
            binary_state = np.where(combined_state >= 0, 1, -1)
            energy_grid[i, j] = hopfield_energy(W, binary_state)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x_vals, y_vals, energy_grid, levels=30)
    plt.colorbar(contour, label="Energy")

    plt.xlabel("Direction of target face")
    plt.ylabel("Direction of another stored face")
    plt.title(title)

    # Mark approximate locations of the two memories
    plt.scatter(1, 0, marker="x", s=100, label="Target face direction")
    plt.scatter(0, 1, marker="o", s=80, label="Other face direction")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energy_landscape_3d_smooth(
    W,
    pattern_a,
    pattern_b,
    title="Smooth 3D Projected Hopfield Energy Landscape"
):
    """
    Smooth 3D projection of the Hopfield energy landscape.

    This uses tanh() instead of sign() so the plotted surface
    looks smoother.
    """

    x_vals = np.linspace(-2, 2, 100)
    y_vals = np.linspace(-2, 2, 100)

    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            combined_state = X[i, j] * pattern_a + Y[i, j] * pattern_b
            smooth_state = np.tanh(combined_state)

            Z[i, j] = hopfield_energy(W, smooth_state)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
        alpha=0.95
    )

    ax.set_xlabel("Direction of target face")
    ax.set_ylabel("Direction of another stored face")
    ax.set_zlabel("Energy")
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.6, aspect=12, label="Energy")

    plt.tight_layout()
    plt.show()

# Fetch Faces

print("Downloading and processing Olivetti Faces...")

faces_data = fetch_olivetti_faces(shuffle=False)
images = faces_data.images

# Preprocess faces into bipolar patterns

binary_patterns = []

for img in images:
    thresh = threshold_otsu(img)
    binary_patterns.append(np.where(img > thresh, 1, -1).flatten())

binary_patterns = np.array(binary_patterns)

N = binary_patterns.shape[1]

print(f"\nEach face has {N} neurons.")
print(f"Estimated Hopfield capacity: {0.14 * N:.2f} patterns.")


# Setup parameters for experiments

np.random.seed(42)

random_people = np.random.choice(40, size=4, replace=False)
target_indices = random_people * 10

loading_conditions = [2, 6, 10, 15, 40, 100]

results = []

fig, axes = plt.subplots(4, 8, figsize=(22, 11))

print(f"\nRunning experiments on randomly selected people IDs: {random_people}")

# Run experiments

for row, target_idx in enumerate(target_indices):
    print(f"\nProcessing Subject {row + 1}...")

    target_face = binary_patterns[target_idx]

    # Create the corrupted cue: 50% of pixels forced to black (-1)
    corrupted_cue = np.copy(target_face)
    noise_indices = np.random.choice(N, size=int(0.5 * N), replace=False)
    corrupted_cue[noise_indices] = -1

    # Plot original and corrupted cue
    axes[row, 0].imshow(target_face.reshape(64, 64), cmap="Greys_r")
    axes[row, 1].imshow(corrupted_cue.reshape(64, 64), cmap="Greys_r")

    if row == 0:
        axes[row, 0].set_title("Original Target", fontweight="bold", fontsize=12)
        axes[row, 1].set_title("Cue\n50% Missing", fontweight="bold", fontsize=12)

    for col, count in enumerate(loading_conditions):

        # Build training set
        other_faces = np.delete(binary_patterns, target_idx, axis=0)
        np.random.shuffle(other_faces)

        training_set = np.vstack([
            target_face,
            other_faces[:count - 1]
        ])

        # Train Hopfield Network
        W = train_hopfield(training_set)

        # Recall from corrupted cue
        recall, energy_history, epochs_used = recall_async(
            W,
            corrupted_cue,
            epochs=10,
            track_energy=True
        )

        # Metrics
        acc = pixel_accuracy(target_face, recall)
        ham = hamming_distance(target_face, recall)
        initial_energy = energy_history[0]
        final_energy = energy_history[-1]
        energy_change = final_energy - initial_energy
        load_ratio = pattern_load_ratio(count, N)
        cap_percent = capacity_percentage(count, N)

        results.append({
            "subject": row + 1,
            "stored_faces": count,
            "neurons": N,
            "pixel_accuracy": acc,
            "hamming_distance": ham,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": energy_change,
            "epochs_used": epochs_used,
            "pattern_load_ratio": load_ratio,
            "capacity_percentage": cap_percent
        })

        print(
            f"{count} faces | "
            f"Accuracy: {acc:.2f}% | "
            f"Hamming: {ham} | "
            f"Energy change: {energy_change:.2f} | "
            f"Epochs: {epochs_used}"
        )

        # Plot recalled image
        axes[row, col + 2].imshow(recall.reshape(64, 64), cmap="Greys_r")

        if row == 0:
            axes[row, col + 2].set_title(
                f"Recall\n{count} Faces",
                fontweight="bold",
                fontsize=12
            )

# Format Recall Grid

for ax in axes.flatten():
    ax.axis("off")

plt.suptitle(
    "Hopfield Network Recall under Increasing Memory Load",
    y=0.98,
    fontsize=18
)

plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

# Save and display results

results_df = pd.DataFrame(results)

print("\nResults table:")
print(results_df)

results_df.to_csv("hopfield_results.csv", index=False)

print("\nSaved results to hopfield_results.csv")

# Plot Metrics

plt.figure(figsize=(8, 5))

for subject in results_df["subject"].unique():
    subject_data = results_df[results_df["subject"] == subject]
    plt.plot(
        subject_data["stored_faces"],
        subject_data["pixel_accuracy"],
        marker="o",
        label=f"Subject {subject}"
    )

plt.xlabel("Number of Stored Faces")
plt.ylabel("Pixel Accuracy (%)")
plt.title("Recall Accuracy vs Number of Stored Faces")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

for subject in results_df["subject"].unique():
    subject_data = results_df[results_df["subject"] == subject]
    plt.plot(
        subject_data["stored_faces"],
        subject_data["hamming_distance"],
        marker="o",
        label=f"Subject {subject}"
    )

plt.xlabel("Number of Stored Faces")
plt.ylabel("Hamming Distance")
plt.title("Recall Error vs Number of Stored Faces")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

for subject in results_df["subject"].unique():
    subject_data = results_df[results_df["subject"] == subject]
    plt.plot(
        subject_data["stored_faces"],
        subject_data["final_energy"],
        marker="o",
        label=f"Subject {subject}"
    )

plt.xlabel("Number of Stored Faces")
plt.ylabel("Final Energy")
plt.title("Final Hopfield Energy vs Number of Stored Faces")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

for subject in results_df["subject"].unique():
    subject_data = results_df[results_df["subject"] == subject]
    plt.plot(
        subject_data["stored_faces"],
        subject_data["pattern_load_ratio"],
        marker="o",
        label=f"Subject {subject}"
    )

plt.axhline(0.14, linestyle="--", label="Rule-of-thumb capacity limit")
plt.xlabel("Number of Stored Faces")
plt.ylabel("Pattern Load Ratio")
plt.title("Pattern Load Ratio vs Hopfield Capacity Rule")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot one energy trajectory

example_target_idx = target_indices[0]
example_target = binary_patterns[example_target_idx]

example_cue = np.copy(example_target)
noise_indices = np.random.choice(N, size=int(0.5 * N), replace=False)
example_cue[noise_indices] = -1

other_faces = np.delete(binary_patterns, example_target_idx, axis=0)
np.random.shuffle(other_faces)

example_count = 100
example_training_set = np.vstack([
    example_target,
    other_faces[:example_count - 1]
])

example_W = train_hopfield(example_training_set)

example_recall, example_energy_history, example_epochs = recall_async(
    example_W,
    example_cue,
    epochs=10,
    track_energy=True
)

plt.figure(figsize=(7, 5))
plt.plot(range(len(example_energy_history)), example_energy_history, marker="o")
plt.xlabel("Recall Epoch")
plt.ylabel("Energy")
plt.title("Energy Descent during Asynchronous Recall")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualise projected energy landscape

# Use target face and another stored face as two projection directions
pattern_a = example_training_set[0]
pattern_b = example_training_set[1]

plot_energy_landscape(
    example_W,
    pattern_a,
    pattern_b,
    title="Projected Energy Landscape for Hopfield Network"
)

pattern_a = example_training_set[0]
pattern_b = example_training_set[1]

plot_energy_landscape_3d_smooth(
    example_W,
    pattern_a,
    pattern_b,
    title="Smooth Projected Energy Landscape"
)