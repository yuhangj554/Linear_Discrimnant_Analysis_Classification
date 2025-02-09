def determinant(matrix):
    width = len(matrix)
    
    if width == 1:
        return matrix[0][0]
    if width == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for col in range(width):
        sub_matrix = [row[:col] + row[col+1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)
    
    return det

def inverse(matrix, precision = 0.000001):
    width = len(matrix)
    result = [[0 for _ in range(width)] for _ in range(width)]
    mat = matrix
    det = determinant(mat)

    for row in range(width):
        for col in range(width):
            sub_matrix = [(i[:col] + i[col+1:]) for i in (mat[:row]+mat[row+1:])]
            result[col][row] = ((-1)**(col+row)) * determinant(sub_matrix) / det
    
    return result

def solution(matrix, vec):
    inverse_mat = inverse(matrix)
    size = len(vec)
    result = [0 for _ in range(size)]

    for row in range (size):
        for col in range(size):
            result[row] += inverse_mat[row][col] * vec[col]

    return result 

def normalize_vector(vec):
    norm = 0
    for value in vec:
        norm += value ** 2
    norm = norm ** 0.5
    for i in range(len(vec)):
        vec[i] /= norm

if __name__ == '__main__':
    matrix = [[1, 2, 3], [4, 9, 6], [7, 8, 9]]
    matrix1 = [[1,4], [3,6]]
    vec = [5,10,10]
    print(inverse(matrix))  
    print(inverse(matrix1))
    print(solution(matrix, vec))
