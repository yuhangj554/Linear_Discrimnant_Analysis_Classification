from tools import solution, inverse, normalize_vector, inner_product

class LDA:
    def __init__(self, data, type_label, categories = []):
        '''
        * MISSING: Exception Handling of length does not match
        * MISSING: Exception Handling of type in 'categories' is not in 'type_label'
        '''
        if categories == []:
            categories = [[type_label[i]] for i in range(len(type_label)) if type_label[i] not in type_label[:i]]
        self.num_cat = len(categories)
        self.num_var = len(data[0])
        self.organized_data  = [] #3-D array
         # cluster the data set based on the classification rule provided
        for cat_index in range(self.num_cat):
            cat = [data[data_index] for data_index in range(len(type_label)) if type_label[data_index] in categories[cat_index]]
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

    def reset(self, data, type_label, categories = []):
        self.__init__(data, type_label, categories = [])
        

    # compute mean vector in each cluster
    def compute_mean(self):
        for cat_index in range(self.num_cat): 
            cat = self.organized_data[cat_index]
            cat_mean = [0 for _ in range(self.num_var)]
            for var_index in range(self.num_var):
                total = 0
                for data_index in range(len(cat)):
                    total += cat[data_index][var_index]
                cat_mean[var_index] = total / len(cat) 
            self.means.append(cat_mean) 

    # compute within class scatter matrix
    def compute_Sw(self):
        self.Sw_matrix = [[self._compute_Swij(row, column) for column in range(self.num_var)] for row in range(self.num_var)]
        """
        for row in range(len(self.means[0])):
            for column in range(len(self.means[0])):
                self.Sw_matrix[row][column] = self._compute_Swij(row, column) 
        """
    #helper function for clarity
    def _compute_Swij(self, row, column):
        result = 0
        for cat_index in range(self.num_cat):
            cat = self.organized_data[cat_index]
            cat_mean = self.means[cat_index]
            for data_index in range(len(cat)):
               result += (cat[data_index][row] - cat_mean[row])*(cat[data_index][column] - cat_mean[column])
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
        
    # draw projection of each data point on to the discriminat vector    
    def compute_converted_data(self):
        for cat_index in range(self.num_cat):
            self.converted_data.append([inner_product(self.organized_data[cat_index][i], self.discriminant_vector) 
                                        for i in range(len(self.organized_data[cat_index]))])

    def compute_converted_mean(self):
            self.converted_means = [inner_product(self.means[cat_index], self.discriminant_vector)
                                        for cat_index in range(self.num_cat)]
            
    def compute_midpoints(self):
        def _simpleSort(arr):
            arr1 = arr.copy()
            result = []
            for i in range(len(arr1)):
                j = i + 1
                minimum  = arr1[i]
                cat_index = i
                while j < len(arr1):
                    if arr1[j] < minimum:
                        minimum, cat_index = arr1[j], j
                    j += 1 
                result.append([cat_index, minimum])
                temp = arr1[i]
                arr1[i] = arr1[cat_index]
                arr1[cat_index] = temp
            return result
        
        midpoints = _simpleSort(self.converted_means)
        for i in range(len(midpoints)-1):
            self.marks.append([i, (midpoints[i][1]+midpoints[i+1][1])/2])



    def compute_converted_stddev(self):
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
        
    # The instance of this class refers to the input data set
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
    #categories_list = []
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
    obj = LDA(data_l, type_l, categories_list)
    obj.train_discriminant_vector()
    print(obj.Sw_matrix)
    print(obj)
    print(obj.discriminant_vector)
    obj.compute_converted_data()
    obj.compute_converted_mean()
    obj.compute_converted_stddev()
    print(obj.converted_data)
    print(obj.converted_means)
    obj.compute_midpoints()
    print(obj.marks)




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