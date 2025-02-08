class LDA:
    def __init__(self, data, type_label, categories = []):
        if categories == []:
            categories = [t for t in type_label if t not in categories]
        self._organized_data  = [[[]] for _ in range(len(categories))] #3-D array
        self.reorganize_data(data, type_label, categories)

    # cluster the data set based on the classification rule provided
    def reorganize_data(self, data, type_label, categories = []):
        '''
        * MISSING: Exception Handling of length does not match
        * MISSING: Exception Handling of type in 'categories' is not in 'type_label'
        '''
        for cat_index in range(len(categories)):
            cat = [data[data_index] for data_index in range(len(type_label)) if type_label[data_index] in categories[cat_index]]
            self._organized_data[cat_index] = cat        
    
    # The instance of this class refers to the input data set
    def __str__(self):
        result = ''
        for cat_index in range(len(self._organized_data)):
            result += "Categories " + str(cat_index) + ":\n"
            cluster = self._organized_data[cat_index]
            for sample in cluster:
                result += str(sample) +"\n"
            result += "\n"
        return result
    
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
    print(obj)