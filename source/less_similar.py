import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from skimage.filters import threshold_otsu


# -----------------------------
# Reproducibility
# -----------------------------

SEED = 42
np.random.seed(SEED)


# -----------------------------
# Hopfield Network Functions
# -----------------------------

def train_hopfield(patterns):
    """
    Hebbian learning rule to calculate the Hopfield weight matrix.
    """
    num_patterns, num_neurons = patterns.shape
    W = np.zeros((num_neurons, num_neurons))

    print(f"{num_patterns} patterns with {num_neurons} neurons each.")

    for p in patterns:
        W += np.outer(p, p)

    np.fill_diagonal(W, 0)
    return W / num_neurons


def hopfield_energy(W, state):
    """
    Calculate Hopfield energy:
    E = -0.5 * s^T W s
    """
    return -0.5 * state @ W @ state


def recall_async(W, pattern, epochs=10, track_energy=True):
    """
    Asynchronous recall.
    Neurons are updated one by one in random order.
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

        if np.array_equal(previous_state, current_state):
            break

    epochs_used = epoch + 1

    return current_state, energy_history, epochs_used


# -----------------------------
# Evaluation Metrics
# -----------------------------

def pixel_accuracy(original, recalled):
    """
    Percentage of pixels that match the original target pattern.
    """
    return np.mean(original == recalled) * 100


def hamming_distance(original, recalled):
    """
    Number of pixels that differ from the original target pattern.
    """
    return np.sum(original != recalled)


def pattern_load_ratio(num_patterns, num_neurons):
    """
    Pattern load ratio:
    alpha = P / N
    """
    return num_patterns / num_neurons


def capacity_percentage(num_patterns, num_neurons):
    """
    Percentage of theoretical Hopfield capacity used.
    """
    estimated_capacity = 0.14 * num_neurons
    return (num_patterns / estimated_capacity) * 100


# -----------------------------
# Similarity-Based Face Selection
# -----------------------------

def face_similarity(face_a, face_b):
    """
    Normalised dot-product similarity between two bipolar face patterns.

    Since faces are represented using -1 and +1, this measures
    how similar their pixel structures are.

    Higher value = more similar.
    Lower value = less similar.
    """
    return np.dot(face_a, face_b) / len(face_a)


def get_least_similar_faces(binary_patterns, target_idx, count):
    """
    Build a training set containing:
    - the target face
    - the least similar faces to the target

    This is used to reduce interference between stored patterns.
    """
    target_face = binary_patterns[target_idx]

    similarities = []

    for idx, face in enumerate(binary_patterns):
        if idx != target_idx:
            sim = face_similarity(target_face, face)
            similarities.append((idx, sim))

    # Sort from least similar to most similar
    similarities.sort(key=lambda item: item[1])

    selected_indices = [idx for idx, sim in similarities[:count - 1]]

    training_set = np.vstack([
        target_face,
        binary_patterns[selected_indices]
    ])

    selected_similarities = [sim for idx, sim in similarities[:count - 1]]

    return training_set, selected_indices, selected_similarities


# -----------------------------
# Energy Landscape Visualisation
# -----------------------------

def plot_energy_landscape_3d_smooth(
    W,
    pattern_a,
    pattern_b,
    title="Smooth 3D Projected Hopfield Energy Landscape"
):
    """
    Smooth 3D projection of the Hopfield energy landscape.

    The true landscape is 4096-dimensional, so this is only a
    2D projection using two stored face directions.

    tanh() is used only for smoothing the visualisation.
    The actual Hopfield recall still uses binary states.
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
    ax.set_ylabel("Direction of least similar stored face")
    ax.set_zlabel("Energy")
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.6, aspect=12, label="Energy")

    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig("least_similar_energy_landscape_3d.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# -----------------------------
# 1. Fetch and Preprocess Faces
# -----------------------------

print("Downloading and processing Olivetti Faces...")

faces_data = fetch_olivetti_faces(shuffle=False)
images = faces_data.images

binary_patterns = []

for img in images:
    thresh = threshold_otsu(img)
    binary_pattern = np.where(img > thresh, 1, -1).flatten()
    binary_patterns.append(binary_pattern)

binary_patterns = np.array(binary_patterns)

N = binary_patterns.shape[1]

print(f"\nEach face has {N} neurons.")
print(f"Estimated Hopfield capacity: {0.14 * N:.2f} patterns.")


# -----------------------------
# 2. Experiment Parameters
# -----------------------------

# Select 4 random people as target subjects
random_people = np.random.choice(40, size=4, replace=False)
target_indices = random_people * 10

loading_conditions = [2, 6, 10, 15, 40, 100]

CORRUPTION_LEVEL = 0.5
MAX_EPOCHS = 10

results = []

fig, axes = plt.subplots(4, 8, figsize=(22, 11))

print(f"\nRunning least-similar-face experiment.")
print(f"Randomly selected target people IDs: {random_people}")
print(f"Seed: {SEED}")


# -----------------------------
# 3. Run Main Experiment
# -----------------------------

for row, target_idx in enumerate(target_indices):
    print(f"\nProcessing Subject {row + 1}...")
    print(f"Target index: {target_idx}")

    target_face = binary_patterns[target_idx]

    # Create corrupted cue
    corrupted_cue = np.copy(target_face)
    noise_indices = np.random.choice(
        N,
        size=int(CORRUPTION_LEVEL * N),
        replace=False
    )
    corrupted_cue[noise_indices] = -1

    # Plot original and cue
    axes[row, 0].imshow(target_face.reshape(64, 64), cmap="Greys_r")
    axes[row, 1].imshow(corrupted_cue.reshape(64, 64), cmap="Greys_r")

    if row == 0:
        axes[row, 0].set_title("Original Target", fontweight="bold", fontsize=12)
        axes[row, 1].set_title("Cue\n50% Missing", fontweight="bold", fontsize=12)

    for col, count in enumerate(loading_conditions):

        # Select least similar faces instead of random background faces
        training_set, selected_indices, selected_similarities = get_least_similar_faces(
            binary_patterns,
            target_idx,
            count
        )

        mean_similarity = np.mean(selected_similarities) if selected_similarities else 0
        max_similarity = np.max(selected_similarities) if selected_similarities else 0

        # Train Hopfield Network
        W = train_hopfield(training_set)

        # Recall from corrupted cue
        recall, energy_history, epochs_used = recall_async(
            W,
            corrupted_cue,
            epochs=MAX_EPOCHS,
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
            "seed": SEED,
            "selection_method": "least_similar_faces",
            "subject": row + 1,
            "target_person_id": int(random_people[row]),
            "target_index": int(target_idx),
            "stored_faces": count,
            "neurons": N,
            "corruption_level": CORRUPTION_LEVEL,
            "pixel_accuracy": acc,
            "hamming_distance": ham,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": energy_change,
            "epochs_used": epochs_used,
            "pattern_load_ratio": load_ratio,
            "capacity_percentage": cap_percent,
            "mean_selected_similarity": mean_similarity,
            "max_selected_similarity": max_similarity,
            "selected_face_indices": selected_indices
        })

        print(
            f"{count} faces | "
            f"Accuracy: {acc:.2f}% | "
            f"Hamming: {ham} | "
            f"Energy change: {energy_change:.2f} | "
            f"Mean similarity: {mean_similarity:.4f} | "
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


# -----------------------------
# 4. Format and Save Recall Grid
# -----------------------------

for ax in axes.flatten():
    ax.axis("off")

plt.suptitle(
    "Hopfield Recall Using Least Similar Stored Faces",
    y=0.98,
    fontsize=18
)

plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig("least_similar_face_recall_grid.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# -----------------------------
# 5. Save Results Table
# -----------------------------

results_df = pd.DataFrame(results)

print("\nResults table:")
print(results_df)

results_df.to_csv("least_similar_hopfield_results.csv", index=False)

print("\nSaved results to least_similar_hopfield_results.csv")


# -----------------------------
# 6. Plot Metric Graphs
# -----------------------------

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
plt.title("Recall Accuracy vs Memory Load\nLeast Similar Face Selection")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("least_similar_recall_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


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
plt.title("Recall Error vs Memory Load\nLeast Similar Face Selection")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("least_similar_recall_error.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


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
plt.title("Final Energy vs Memory Load\nLeast Similar Face Selection")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("least_similar_final_energy.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


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
plt.savefig("least_similar_pattern_load_ratio.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# -----------------------------
# 7. Plot One Energy Trajectory
# -----------------------------

example_target_idx = target_indices[0]
example_target = binary_patterns[example_target_idx]

example_cue = np.copy(example_target)
noise_indices = np.random.choice(
    N,
    size=int(CORRUPTION_LEVEL * N),
    replace=False
)
example_cue[noise_indices] = -1

example_count = 100

example_training_set, example_selected_indices, example_similarities = get_least_similar_faces(
    binary_patterns,
    example_target_idx,
    example_count
)

example_W = train_hopfield(example_training_set)

example_recall, example_energy_history, example_epochs = recall_async(
    example_W,
    example_cue,
    epochs=MAX_EPOCHS,
    track_energy=True
)

plt.figure(figsize=(7, 5))
plt.plot(range(len(example_energy_history)), example_energy_history, marker="o")
plt.xlabel("Recall Epoch")
plt.ylabel("Energy")
plt.title("Energy Descent during Asynchronous Recall\nLeast Similar Face Selection")
plt.grid(True)
plt.tight_layout()
plt.savefig("least_similar_energy_descent.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# -----------------------------
# 8. Visualise Smooth 3D Projected Energy Landscape
# -----------------------------

pattern_a = example_training_set[0]
pattern_b = example_training_set[1]

plot_energy_landscape_3d_smooth(
    example_W,
    pattern_a,
    pattern_b,
    title="Smooth Projected Energy Landscape\nLeast Similar Face Selection"
)