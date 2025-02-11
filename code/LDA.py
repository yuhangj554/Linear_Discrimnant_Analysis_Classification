import numpy as np
from tools import solution, inverse, normalize_vector, inner_product, multiply_by_scalar

class LDA:
    def __init__(self, data, type_label, categories = []):
        '''
        * MISSING: Exception Handling of length does not match
        * MISSING: Exception Handling of type in 'categories' is not in 'type_label'
        '''
        self.categories = categories
        if categories == []:
            self.categories = [[type_label[i]] for i in range(len(type_label)) if type_label[i] not in type_label[:i]]
        self.num_cat = len(self.categories)
        self.num_var = len(data[0])
        self.organized_data  = [] #3-D array
         # cluster the data set based on the classification rule provided
        for cat_index in range(self.num_cat):
            cat = [data[data_index] for data_index in range(len(type_label)) if type_label[data_index] in self.categories[cat_index]]
            self.organized_data.append(cat)

        self.means = []
        self.Sw_matrix = []
        self.Sb_matrix = []
        self.catdiff = []
        self.discriminant_vector = []

        self.converted_data = []
        self.converted_means = []
        self.converted_stddev = []

        self.marks = []
        self.edge_category = 0
        self.classify_functions = []
        self.classify_constants = []
        self.isStddev = False
        self.result_table = []

    def reset(self, data, type_label, categories = []):
        self.__init__(data, type_label, categories = [])
        

    # compute mean vector in each cluster
    def compute_mean(self):
        self.means = []
        self.overall_mean = [0 for _ in range(self.num_var)]
        for cat_index in range(self.num_cat): 
            cat = self.organized_data[cat_index]
            cat_mean = [0 for _ in range(self.num_var)]
            for var_index in range(self.num_var):
                total = 0
                for data_index in range(len(cat)):
                    total += cat[data_index][var_index]
                self.overall_mean[var_index] += total
                cat_mean[var_index] = total / len(cat) 
            self.means.append(cat_mean) 
        self.overall_mean = multiply_by_scalar(self.overall_mean, 1/len(self.organized_data))
        

    # compute within class scatter matrix
    def compute_Sw(self):
        self.Sw_matrix = [[self._compute_Swij(row, column) for column in range(self.num_var)] for row in range(self.num_var)]
    #helper function for clarity
    def _compute_Swij(self, row, column):
        result = 0
        for cat_index in range(self.num_cat):
            cat = self.organized_data[cat_index]
            cat_mean = self.means[cat_index]
            for data_index in range(len(cat)):
               result += (cat[data_index][row] - cat_mean[row])*(cat[data_index][column] - cat_mean[column])
        return result

    # compute between class scatter matrix
    def compute_Sb(self):
        self.Sb_matrix = [[self._compute_Sbij(row, column) for column in range(self.num_var)] for row in range(self.num_var)]
    #helper function for clarity
    def _compute_Sbij(self, row, column):
        result = 0
        for cat_index in range(self.num_cat):
            cat_mean = self.means[cat_index]
            result += (cat_mean[row] - self.overall_mean[row]) * (cat_mean[column] - self.overall_mean[column])
            
        return result
    
    # compute the distance in means of the two categories
    # only used when there are only two categories
    def compute_catdiff(self):
        self.catdiff = [self.means[0][i] - self.means[1][i] for i in range(self.num_var)]
    
    # train the discriminant vector for the dataset
    # the projection of dataset onto this vector has greatest between class scatter and smallest within class scatter 
    def train_discriminant_vector(self):
        if self.num_cat == 2:
            self.compute_mean()
            self.compute_Sw()
            self.compute_catdiff()
            self.discriminant_vector = solution(self.Sw_matrix, self.catdiff)
            normalize_vector(self.discriminant_vector)
        else:
            self.compute_mean()
            self.compute_Sw()
            self.compute_Sb()
            inverse_Sw = inverse(self.Sw_matrix)
            result_matrix = []
            for row in range(self.num_var):
                result_matrix.append([])
                for col in range(self.num_var):
                    vec = [self.Sb_matrix[i][col] for i in range(self.num_var)]
                    result_matrix[row].append(inner_product(inverse_Sw[row], vec))
            eigenvalues, eigenvectors = np.linalg.eig(np.array(result_matrix))
            max_index = np.argmax(eigenvalues)
            self.discriminant_vector = eigenvectors.real[:, max_index].tolist()

        
    # draw projection of each data point on to the discriminat vector    
    def compute_converted_data(self):
        self.converted_data = []
        for cat_index in range(self.num_cat):
            self.converted_data.append([inner_product(self.organized_data[cat_index][i], self.discriminant_vector) 
                                        for i in range(len(self.organized_data[cat_index]))])

    def compute_converted_mean(self):
            self.converted_means = [inner_product(self.means[cat_index], self.discriminant_vector)
                                        for cat_index in range(self.num_cat)]
            
    def compute_midpoints(self):
        self.marks = []
        def _simpleSort(arr):
            indexed_arr = [[i, arr[i]] for i in range(len(arr))]
            for i in range(len(arr)):
                j = i + 1
                minimum =  indexed_arr[i][1]
                pos = i
                while j < len(arr):
                    if indexed_arr[j][1] < minimum:
                        minimum = indexed_arr[j][1]
                        pos = j
                    j += 1
                temp = indexed_arr[i]
                indexed_arr[i] = indexed_arr[pos]
                indexed_arr[pos] = temp
            return indexed_arr

        ordered_means = _simpleSort(self.converted_means)
        self.edge_category = ordered_means[-1][0]
        for i in range(len(ordered_means)-1):
            self.marks.append([ordered_means[i][0], (ordered_means[i][1]+ordered_means[i+1][1])/2])

    def compute_converted_stddev(self):
        self.converted_stddev = []
        for cat_index in range(self.num_cat):
            deviation = 0
            for data_index in range(len(self.converted_data[cat_index])):
                deviation += (self.converted_data[cat_index][data_index] - self.converted_means[cat_index]) ** 2
            deviation /= (len(self.converted_data[cat_index]) - 1) if len(self.converted_data[cat_index]) > 1 else 1
            deviation = deviation ** 0.5
            self.converted_stddev.append(deviation)
        
        # edge cases where the stddev withn one or more categories is 0
        minimum = 1
        count = 2
        for cat_index in range(self.num_cat):
            if self.converted_stddev[cat_index] != 0:
                minimum = self.converted_stddev[cat_index]
                count  = len(self.converted_data[cat_index])
        for cat_index in range(self.num_cat):
            if self.converted_stddev[cat_index] == 0:
                continue
            if self.converted_stddev[cat_index] < minimum:
                minimum = self.converted_stddev[cat_index]
                count = len(self.converted_data[cat_index])
        for cat_index in range(self.num_cat):
            if self.converted_stddev[cat_index] == 0:
                self.converted_stddev[cat_index] = (minimum / count) ** (1/len(self.converted_data[cat_index]))
    
    def compute_classify_functions(self):
        normalize_vector(self.converted_stddev)
        self.classify_functions = [multiply_by_scalar(self.discriminant_vector, 1/self.converted_stddev[i]) for i in range(self.num_cat)]
        self.classify_constants = [-self.converted_means[i] / self.converted_stddev[i] for i in range(self.num_cat)]

    def classify_all(self, isStddev = False):
        if self.discriminant_vector == []:
            self.train_discriminant_vector()
        self.isStddev = isStddev
        self.compute_converted_data()
        self.compute_converted_mean()
        
        if isStddev == False:
            self.compute_midpoints()
        else:
            self.compute_converted_stddev()
            self.compute_classify_functions()
        
        for cat_index in range(len(self.organized_data)):
            cat_result = []
            for data_index in range(len(self.organized_data[cat_index])):
                cat_result.append(self.classify_single(self.organized_data[cat_index][data_index]))
            self.result_table.append(cat_result)

    
    def classify_single(self, data):
        if self.isStddev == False:
            converted_data = inner_product(data, self.discriminant_vector)
            result = self.edge_category
            for i in range(self.num_cat-1):
                if converted_data < self.marks[i][1]:
                    result =  self.marks[i][0]
                    return result
            return result
        else:
            zscores = [abs(inner_product(data, self.classify_functions[i]) + self.classify_constants[i]) for i in range(self.num_cat)]
            result = zscores.index(min(zscores))
            return result
        
    def Classification_Result(self, show_details = False, file = None):
        correct_count = [0 for _ in range(self.num_cat)]

        if show_details:
            print("Detailed Classification of Every Data Point: ")
            for cat_index in range(self.num_cat):
                print("Category " + str(cat_index) + ": (includes types: " + str(self.categories[cat_index]) + ")")
                cat = self.organized_data[cat_index]
                cat_result = self.result_table[cat_index]
                for data_index in range(len(cat)):
                    if cat_result[data_index] == cat_index:
                        judgement = True
                        correct_count[cat_index] += 1
                    else:
                        judgement = False
                    print(str(cat[data_index])+"\t"+str(cat_result[data_index])+"\t"+str(judgement))
                print()
        else:
            for cat_index in range(self.num_cat):
                for data_index in range(len(self.organized_data[cat_index])):
                    if self.result_table[cat_index][data_index] == cat_index:
                        correct_count[cat_index] += 1

        accuracy_list = []
        total_count = 0
        for cat_index in range(self.num_cat):
            accuracy_list.append(correct_count[cat_index]/len(self.organized_data[cat_index]))
            total_count += len(self.organized_data[cat_index])
        overall_accuracy = inner_product(correct_count, [1 for _ in range(self.num_cat)]) / total_count

        print("Accuracy: ")
        for cat_index in range(self.num_cat):
            print("Category " + str(cat_index) + ": (includes types: " + str(self.categories[cat_index]) + ")")
            print("Correct / total:\t" + str(correct_count[cat_index]) + " / " + str(len(self.organized_data[cat_index])))
            print(str(round(accuracy_list[cat_index], 4)*100) +  "%")
            print()
        print("Overall Accuracy: " + str(round(overall_accuracy, 4)*100) +  "%")
        print()

        if not self.isStddev:
            print("Classification Rules: Midpoints")
            print("Discriminant Vector:")
            print("\t", end="")
            for var_index in range(self.num_var):
                print(round(self.discriminant_vector[var_index], 4), "X" + str(var_index+1), end="")
                if var_index < self.num_var-1:
                    print(" + ", end="")
            print()
            print()
            print("Midpoints(Boundaries):")
            print("\t", end="")
            for i in range(len(self.marks)):
                print("_______"+str(round(self.marks[i][1],4)), end="")
            print("_______\n\t",end="")
            for i in range(len(self.marks)):
                print("   "+str(self.marks[i][0])+"         ", end="")
                if (self.marks[i][1] < 0):
                    print(" ", end="")
            print("   "+str(self.edge_category))
            print("\n" +"Plug a data point into the discriminant vector. Its Category is determined by the area it falls into.")
        else:
            print("Classification Rules: Z-scores")
            print("Discriminant Vector:")
            print("\t", end="")
            for var_index in range(self.num_var):
                print(round(self.discriminant_vector[var_index], 4), "X" + str(var_index+1), end="")
                if var_index < self.num_var-1:
                    print(" + ", end="")
            print()
            print()
            print("Discriminant Functions: ")
            for cat_index in range(self.num_cat):
                print("\t", end="")
                for var_index in range(self.num_var):
                    print(round(self.classify_functions[cat_index][var_index], 4), "X" + str(var_index+1), end="")
                    print(" + ", end="")
                print(round(self.classify_constants[cat_index], 4))
            print()
            print("Plug a data point into each of the discriminant function. It would be classified to the category with the smallest function value.")

        
    # print the organized data set when called as str
    def __str__(self):
        result = ''
        for cat_index in range(self.num_cat):
            result += "Categories " + str(cat_index) + ":\n"
            cluster = self.organized_data[cat_index]
            for sample in cluster:
                result += str(sample) +"\n"
            # result += "Mean Vector of this cluster:\n" + str(self.means[cat_index]) +"\n"
            result += "\n"
        return result


if __name__ == "__main__":
    categories_list = [['Y', 'I'],  ['K', 'Q']]
    categories_list = []
    type_l = ['Y','K', 'K', 'Y', 'K', 'Q', 'Q', 'I']
    data_l = [[1,2,3,4],
              [2,2,3,4],
              [4,7,6,8], 
              [9,6,1,0],
              [2,7,3,0],
              [8,8,4,8],
              [9,6,1,9],             
              [1,0,1,0]
              ]
    type_l = ["this", "that"]
    data_l = [
        [1,2],
        [2,1]
    ]
    obj = LDA(data_l, type_l, categories_list)
    '''
    obj.compute_mean()
    obj.compute_Sb()
    print(inverse(obj.Sb_matrix))
    obj.train_discriminant_vector()
    print(obj)
    print(obj.discriminant_vector)
    obj.compute_converted_data()
    obj.compute_converted_mean()
    obj.compute_converted_stddev()
    print(obj.converted_data)
    print(obj.converted_means)
    print(obj.converted_stddev)

    obj.classify_all(False)
    print("functions: "+str(obj.classify_functions))
    print("constants:"+str(obj.classify_constants))
    print(obj.result_table)

    print("Marks:\n" + str(obj.marks))
    print("converted data:\n" + str(obj.converted_data))
    '''
    obj.classify_all(True)
    obj.Classification_Result(True)


'''
if __name__ == "__main__":
    categories_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    type_l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    data_l = [[1,5], [1,5], [1,5], [1,5], [1,5], [1,5], [1,5], [1,5], [1,5], [-1,-5]
              , [1,5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5]]
    type2 = [1,1,2,2]
    categories2 = [[1],[2]]
    data2 = [[0,1], [1,1], [0, -1], [-1,-1]]
    obj = LDA(data_l, type_l, categories_list)
    obj2 = LDA(data2, type2, categories2)
    obj.train_discriminant_vector()
    obj2.train_discriminant_vector()
    obj2.compute_converted_data()
    obj2.compute_converted_mean()
    obj2.compute_converted_stddev()
    print(obj2)
    print(obj2.discriminant_vector)
    print(obj2.converted_data)
    print(obj2.converted_means)
    print(obj2.converted_stddev)
    obj2.compute_midpoints()
    print(obj2.marks)
''' 