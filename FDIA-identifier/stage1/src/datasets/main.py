from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .anomaly import AnomalyDataset

def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('anomaly')
    assert dataset_name in implemented_datasets

    dataset = None
    
    if dataset_name == 'anomaly':
        dataset = AnomalyDataset(root='../data/generated/118-bus', normal_class=0)

    return dataset
