import csv
import LDA

print("Enter the path of the file: ('m' for manual data entry)")
path = input() # ../data/Female_lizards.csv
if path != "m":
    with open(path.strip(), 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)

        print("\nEnter the variables you wish to use, separate by comma: ('!' to ignore)")
        print("All variables: "+str(header[1:]))
        temp = input()
        if temp.strip() != "!":
            includes_list = temp.split(",")
            for i in range(len(includes_list)):
                includes_list[i] = includes_list[i].strip()
                includes_list[i] =  includes_list[i][1:] if includes_list[i][0] == "\'" else includes_list[i]
                includes_list[i] =  includes_list[i][:-1] if includes_list[i][-1] == "\'" else includes_list[i]
                if includes_list[i] not in header[1:]:
                    raise ValueError(str(includes_list[i])+" is not in the header of the input file.")
        else:
            includes_list = header[1:]
        
        data = []
        type_labels = []
        all_types = set()
        for row in reader:
            data.append([float(value) for i, value in enumerate(row) if header[i] in includes_list])
            type_labels.append(row[0])
            all_types.add(row[0])

    print("\nEnter the number of categories you wish to classify the dataset into, at least 2: ('!' to ignore)")
    temp = input()
    categories = []
    if temp != "!":
        num_cat = int(temp)
        if num_cat < 2:
            raise ValueError(str(num_cat)+"is not valid for the number of categories")
        print("All types: " + str(all_types))
        for cat_index in range(num_cat):
            print("Enter the types you wish to be classified as Category " + str(cat_index)  + ", separated by comma: ")
            includes_list = input().split(",")
            for i in range(len(includes_list)):
                includes_list[i] = includes_list[i].strip()
                includes_list[i] =  includes_list[i][1:] if includes_list[i][0] == "\'" else includes_list[i]
                includes_list[i] =  includes_list[i][:-1] if includes_list[i][-1] == "\'" else includes_list[i]
                if includes_list[i] not in all_types:
                    raise ValueError(str(includes_list[i])+" is not in the types listed in the input file.")
                categories.append(includes_list)
        print(categories)
        
 
