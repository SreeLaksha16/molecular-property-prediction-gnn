import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Load test predictions
# ==========================================

df = pd.read_csv("test_predictions.csv")

print("Columns found in CSV:")
print(df.columns.tolist())

print()
print("Number of predictions:", len(df))


# ==========================================
# 2. Detect columns
# ==========================================

actual_column = None
predicted_column = None

for column in df.columns:

    name = column.lower()

    if "actual" in name or "target" in name:
        actual_column = column

    if "pred" in name:
        predicted_column = column


if actual_column is None or predicted_column is None:

    print()
    print("Could not automatically identify columns.")
    print("Please check test_predictions.csv.")

    print(df.head())

    raise SystemExit


actual = df[actual_column]
predicted = df[predicted_column]


# ==========================================
# 3. Actual vs Predicted Plot
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.7
)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual ESOL Value")
plt.ylabel("Predicted ESOL Value")

plt.title("Actual vs Predicted ESOL Values")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.close()

print()
print("Saved:")
print("actual_vs_predicted.png")


# ==========================================
# 4. Prediction Error
# ==========================================

errors = predicted - actual


plt.figure(figsize=(8, 6))

plt.hist(
    errors,
    bins=20
)

plt.xlabel("Prediction Error")

plt.ylabel("Number of Molecules")

plt.title("Distribution of Prediction Errors")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "prediction_error_distribution.png",
    dpi=300
)

plt.close()

print("Saved:")
print("prediction_error_distribution.png")


# ==========================================
# 5. Finish
# ==========================================

print()
print("========================================")
print("PLOTTING COMPLETE")
print("========================================")

print("Created files:")
print("1. actual_vs_predicted.png")
print("2. prediction_error_distribution.png")