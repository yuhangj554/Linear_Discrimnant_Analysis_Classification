from tools import solution, inverse, normalize_vector

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

        self.catdiff = []
        self.discriminant_vector = []

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
    
    
    def compute_discriminant_vector(self):
        if self.num_cat == 2:
            self.compute_mean()
            self.compute_Sw()
            self.compute_catdiff()
            self.discriminant_vector = solution(self.Sw_matrix, self.catdiff)
            normalize_vector(self.discriminant_vector)
        

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

'''    
if __name__ == "__main__":
    categories_list = [['Y', 'I'],  ['K', 'Q']]
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
    obj.compute_discriminant_vector()
    print(obj)
    print(obj.Sw_matrix)
    print(obj.catdiff)
    print(obj.discriminant_vector)
'''

if __name__ == "__main__":
    categories_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    type_l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    data_l = [[1,5], [1,5], [1,5], [1,5], [1,5], [1,5], [1,5], [1,5], [1,5], [-1,-5]
              , [1,5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5], [-1,-5]]
    type2 = [1,1,2,2]
    categories2 = [[1],[2]]
    data2 = [[0,1], [3,4], [0, -1], [7,11]]
    obj = LDA(data_l, type_l, categories_list)
    obj2 = LDA(data2, type2, categories2)
    obj.compute_discriminant_vector()
    obj2.compute_mean()
    obj2.compute_Sw()
    obj2.compute_discriminant_vector()
    print(obj2)
    print(obj2.discriminant_vector)
