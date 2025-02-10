import csv
import LDA

print("Enter the path of the file: ('m' for manual data entry)")
path = input()
if path != "m":
    with open(path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        header.pop(0)
        print(header)

