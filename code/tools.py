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
    if width <= 7:
        result = [[0 for _ in range(width)] for _ in range(width)]
        det = determinant(matrix)

        if width == 1:
            result = [[1/matrix[0][0]]]
            return result


        if det == 0:
            matrix[0][0] += precision
            for row in range(width):
                for col in range(width):
                    matrix[row][col] += precision
            det = determinant(matrix)
        
        for row in range(width):
            for col in range(width):
                sub_matrix = [(i[:col] + i[col+1:]) for i in (matrix[:row]+matrix[row+1:])]
                result[col][row] = ((-1)**(col+row)) * determinant(sub_matrix) / det
        
        return result
    else:
        i_mat = [[1 if row == col else 0 for col in range(width)] 
                for row in range(width)
        ]
        for row in range(width):
            for row_e in range(width):
                if row == row_e:
                    continue
                if matrix[row][row] == 0:
                   matrix[row][row] += precision
                multiple = matrix[row_e][row] / matrix[row][row]
                for col in range(width):
                    matrix[row_e][col] -= multiple*matrix[row][col]
                    i_mat[row_e][col] -= multiple*i_mat[row][col]
                coeff = matrix[row][row]
                for col in range(width):
                    matrix[row][col] /= coeff
                    i_mat[row][col] /= coeff
        return i_mat

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

def inner_product(vec1, vec2):
    result = 0
    for i in range (len(vec1)):
        result += vec1[i] * vec2[i]
    return result

def multiply_by_scalar(vec, scalar):
    result = []
    for i in range(len(vec)):
        result.append(vec[i] * scalar)
    return result

if __name__ == '__main__':
    matrix = [[1, 2, 3], [4, 9, 6], [7, 8, 9]]
    matrix1 = [[1,4], [3,6]]
    vec = [5,10,10]
    print(inverse(matrix))  
    print(inverse(matrix1))
    print(solution(matrix, vec))
