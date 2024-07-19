from gardenpy.utils.data_utils import MNISTFNNDataLoader

# todo: make folder data/mnist/data (because it doesn't exist for some reason) and run data_generator.py in there
# todo: make folder demo/mnist/dnn_mnist/data/processed_data (because it doesn't exist for some reason) and run process.py there
dataloader = MNISTFNNDataLoader(
    root=...,  # todo: input processed data global root
    values_path='values.csv',
    labels_path='labels.csv',
    save_to_memory=True
)

for i in range(5):
    values, labels = next(iter(dataloader))
    print(values)
    print('----------')
    print(labels)
    print('----------')
