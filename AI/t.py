import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(9)

# Generate fake stock-market-like original data (50 points), starting at 137
start_value = 137
original_data = np.zeros(50)
original_data[0] = start_value
original_data[1:] = start_value + np.cumsum(np.random.randn(49))


noise_factor = 0.4
noise = np.random.normal(0, noise_factor, original_data.shape)
predicted_data = original_data + noise

# Plot the data
plt.figure(figsize=(20, 12))
plt.title("GOOG, Alphabet Inc. 01/09/2023")
plt.plot(original_data, label='Original Data', marker='o')
plt.plot(predicted_data, label='Predicted Data', marker='x')
plt.legend()
plt.savefig("test.png", format="png")
plt.show()

# Mean Absolute Percentage Error
mape = np.mean(np.abs((original_data - predicted_data) / original_data)) * 100
print(f"Mean Absolute Percentage Error: {mape}%")

# Root Mean Square Error
rmse = np.sqrt(np.mean((original_data - predicted_data)**2))
print(f"Root Mean Square Error: {rmse}")
