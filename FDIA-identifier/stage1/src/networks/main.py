from .anomaly_net import Anomaly_Net, Anomaly_Net_Autoencoder
from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('anomaly_Net')
    assert net_name in implemented_networks

    net = None

    if net_name == 'anomaly_Net':
        net = Anomaly_Net()
        
    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('anomaly_Net')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'anomaly_Net':
        ae_net = Anomaly_Net_Autoencoder()
    return ae_net
