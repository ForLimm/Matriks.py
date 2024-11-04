import numpy as np

# Set random seed to ensure consistency
np.random.seed(42)

# Step 1: Initialize random matrices A (2x3) and B (3x4)
A = np.random.randint(1, 10, (2, 3))
B = np.random.randint(1, 10, (3, 4))

# Step 2: Matrix multiplication A * B using numpy library
result_lib = np.matmul(A, B)

# Step 3: Matrix multiplication A * B without using numpy library
result_manual = [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(4)] for i in range(2)]

# Step 4: Dot product A . B using numpy library (also np.matmul in this context)
dot_lib = np.dot(A, B)

# Step 5: Dot product A . B without using numpy library (manual calculation)
dot_manual = [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(4)] for i in range(2)]

# Display results
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("\nResult of A * B using library:\n", result_lib)
print("Result of A * B without library:\n", result_manual)
print("\nDot product A . B using library:\n", dot_lib)
print("Dot product A . B without library:\n", dot_manual)
