import pandas as pd
import matplotlib.pyplot as plt

baseline_path = "hopfield_results.csv"
least_similar_path = "least_similar/least_similar_hopfield_results.csv"

baseline_df = pd.read_csv(baseline_path)
least_df = pd.read_csv(least_similar_path)

baseline_df["method"] = "Baseline"
least_df["method"] = "Least similar"

common_cols = [
    "subject",
    "stored_faces",
    "neurons",
    "pixel_accuracy",
    "hamming_distance",
    "initial_energy",
    "final_energy",
    "energy_change",
    "epochs_used",
    "pattern_load_ratio",
    "capacity_percentage",
    "method"
]

compare_df = pd.concat(
    [
        baseline_df[common_cols],
        least_df[common_cols]
    ],
    ignore_index=True
)

plt.figure(figsize=(8, 5))

for method in compare_df["method"].unique():
    temp = compare_df[compare_df["method"] == method]
    mean_acc = temp.groupby("stored_faces")["pixel_accuracy"].mean()
    plt.plot(mean_acc.index, mean_acc.values, marker="o", label=method)

plt.xlabel("Number of Stored Faces")
plt.ylabel("Mean Pixel Accuracy (%)")
plt.title("Mean Recall Accuracy: Baseline vs Least Similar")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_mean_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))

for method in compare_df["method"].unique():
    temp = compare_df[compare_df["method"] == method]
    mean_hd = temp.groupby("stored_faces")["hamming_distance"].mean()
    plt.plot(mean_hd.index, mean_hd.values, marker="o", label=method)

plt.xlabel("Number of Stored Faces")
plt.ylabel("Mean Hamming Distance")
plt.title("Mean Recall Error: Baseline vs Least Similar")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_mean_hamming_distance.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

for subject, ax in zip(sorted(compare_df["subject"].unique()), axes.flatten()):
    subj = compare_df[compare_df["subject"] == subject]

    for method in subj["method"].unique():
        temp = subj[subj["method"] == method]
        ax.plot(
            temp["stored_faces"],
            temp["pixel_accuracy"],
            marker="o",
            label=method
        )

    ax.set_title(f"Subject {subject}")
    ax.set_xlabel("Stored Faces")
    ax.set_ylabel("Pixel Accuracy (%)")
    ax.grid(True)
    ax.legend()

plt.suptitle("Per-Subject Recall Accuracy Comparison", y=1.02)
plt.tight_layout()
plt.savefig("comparison_per_subject_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

for subject, ax in zip(sorted(compare_df["subject"].unique()), axes.flatten()):
    subj = compare_df[compare_df["subject"] == subject]

    for method in subj["method"].unique():
        temp = subj[subj["method"] == method]
        ax.plot(
            temp["stored_faces"],
            temp["hamming_distance"],
            marker="o",
            label=method
        )

    ax.set_title(f"Subject {subject}")
    ax.set_xlabel("Stored Faces")
    ax.set_ylabel("Hamming Distance")
    ax.grid(True)
    ax.legend()

plt.suptitle("Per-Subject Recall Error Comparison", y=1.02)
plt.tight_layout()
plt.savefig("comparison_per_subject_hamming_distance.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


merged = pd.merge(
    baseline_df,
    least_df,
    on=["subject", "stored_faces"],
    suffixes=("_baseline", "_least")
)

merged["accuracy_improvement"] = (
    merged["pixel_accuracy_least"] - merged["pixel_accuracy_baseline"]
)

plt.figure(figsize=(8, 5))

mean_improvement = merged.groupby("stored_faces")["accuracy_improvement"].mean()

plt.axhline(0, linestyle="--", label="No improvement")
plt.plot(mean_improvement.index, mean_improvement.values, marker="o")

plt.xlabel("Number of Stored Faces")
plt.ylabel("Mean Accuracy Improvement (%)")
plt.title("Accuracy Improvement from Least-Similar Selection")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_accuracy_improvement.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


merged["hamming_improvement"] = (
    merged["hamming_distance_baseline"] - merged["hamming_distance_least"]
)

plt.figure(figsize=(8, 5))

mean_hamming_improvement = merged.groupby("stored_faces")["hamming_improvement"].mean()

plt.axhline(0, linestyle="--", label="No improvement")
plt.plot(mean_hamming_improvement.index, mean_hamming_improvement.values, marker="o")

plt.xlabel("Number of Stored Faces")
plt.ylabel("Mean Hamming Improvement")
plt.title("Hamming Distance Improvement from Least-Similar Selection")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_hamming_improvement.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


if "mean_selected_similarity" in least_df.columns:
    plt.figure(figsize=(8, 5))

    for subject in sorted(least_df["subject"].unique()):
        subj = least_df[least_df["subject"] == subject]
        plt.plot(
            subj["mean_selected_similarity"],
            subj["pixel_accuracy"],
            marker="o",
            label=f"Subject {subject}"
        )

    plt.xlabel("Mean Selected Similarity")
    plt.ylabel("Pixel Accuracy (%)")
    plt.title("Least-Similar Experiment: Similarity vs Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("least_similar_similarity_vs_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

summary = compare_df.groupby(["method", "stored_faces"]).agg(
    mean_pixel_accuracy=("pixel_accuracy", "mean"),
    mean_hamming_distance=("hamming_distance", "mean"),
    mean_energy_change=("energy_change", "mean"),
    mean_epochs_used=("epochs_used", "mean")
).reset_index()

print("\nSummary table:")
print(summary)

summary.to_csv("comparison_summary.csv", index=False)

print("\nSaved plots and comparison_summary.csv")