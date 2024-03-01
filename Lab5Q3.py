import numpy as np
import matplotlib.pyplot as plt

# Generate random data points for class0 (Blue)
np.random.seed(0)  # For reproducibility
class0_x = np.random.randint(1, 11, 10)
class0_y = np.random.randint(1, 11, 10)

# Generate random data points for class1 (Red)
class1_x = np.random.randint(1, 11, 10)
class1_y = np.random.randint(1, 11, 10)

# Create scatter plot
plt.scatter(class0_x, class0_y, color='blue', label='Class 0')
plt.scatter(class1_x, class1_y, color='red', label='Class 1')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Training Data')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()
