def transpose_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    transpose = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = matrix[i][j]

    return transpose

matrix = [[34, 5, 87],
          [3, 4, 12],
          [34, 78, 91]]


transpose = transpose_matrix(matrix)


print("Original Matrix:")
for row in matrix:
    print(row)


print("\nTranspose Matrix:")
for row in transpose:
    print(row)