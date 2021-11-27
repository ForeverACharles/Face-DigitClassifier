from read import read_datasets, print_entry, print_dataset
from perceptron import p_train, p_evaluate

def main():
    digits_dataset, faces_dataset = read_datasets()

    #our testing will have us limit the percent of the dataset to perform training on
    digits_dataset = trim_dataset(digits_dataset, 1)

    print_dataset(digits_dataset[0], 0.1)

    p_train(digits_dataset)
    p_evaluate(digits_dataset)

    p_train(faces_dataset)
    p_evaluate(faces_dataset)

def trim_dataset(dataset, percent):
    size = int(len(dataset[0]) * percent)
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:size]

    return dataset

if __name__ == '__main__':
    main()