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

def inverse(matrix):
    width = len(matrix)
    result = [[0 for _ in range(width)] for _ in range(width)]
    det = determinant(matrix)

    for row in range(width):
        for col in range(width):
            sub_matrix = [(i[:col] + i[col+1:]) for i in (matrix[:row]+matrix[row+1:])]
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
    


matrix = [[1, 2, 3], [4, 9, 6], [7, 8, 9]]
matrix1 = [[1,4], [3,6]]
vec = [5,10,10]
print(inverse(matrix))  
print(inverse(matrix1))
print(solution(matrix, vec))
