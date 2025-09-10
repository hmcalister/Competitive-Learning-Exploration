from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

class AbstractDataset(ABC):
    def __init__(self, num_points: int, num_features: int):
        self.num_points = num_points
        self.num_features = num_features

        # The data points of the class, of size (num_points, num_features)
        self.X: np.ndarray = None # type: ignore

        # The classes of the data points, of size (num_points)
        # Each element should be a class index, i.e. the datatype is int
        self.y: np.ndarray = None # type: ignore

class UniformRandomDataset(AbstractDataset):
    def __init__(self, num_points: int, num_features: int):
        super().__init__(num_points, num_features)
        self.X = np.random.normal(np.zeros((self.num_points, self.num_features)), 1)
        self.y = np.zeros(self.num_points)

class PrototypeDataset(AbstractDataset):
    def __init__(self, num_prototypes: int, instance_noise: float, num_points: int, num_features: int):
        super().__init__(num_points, num_features)
        self.num_prototypes = num_prototypes
        
        # Shape = (self.num_prototypes, self.num_features)
        self.prototypes = np.random.normal(np.zeros((self.num_prototypes, self.num_features)), 1)

        noise_vectors = np.random.normal(np.zeros((self.num_points, self.num_features)), instance_noise)
        
        # Repeat the prototypes until we exceed the number of points, then truncate to num points
        self.X = np.tile(self.prototypes, [self.num_points//self.num_prototypes + 1, 1])[:num_points, :]

        self.X = self.X + noise_vectors
        self.y = np.tile(np.arange(num_prototypes), self.num_points//self.num_prototypes + 1)[:num_points]

    def visualize(self):
        if self.num_features != 2:
            return

        X  = self.X
        y = self.y
        for prototype_index in range(self.num_prototypes):
            prototype_mask = y==prototype_index
            plt.scatter(X[prototype_mask, 0], X[prototype_mask, 1], label=f"{prototype_index}")
        plt.legend(title="Prototype Index")
        plt.show()

class MNISTDataset(AbstractDataset):
    def __init__(self, num_points: int = 70000, data_home="datasets/mnist"):
        """
        Create a new MNIST Dataset.

        num_points determines how many images to keep (chosen randomly)
        num_features is always 784 (28*28).
        """
        super().__init__(num_points, 28*28)
        mnist = fetch_openml('mnist_784', data_home=data_home)
        self.X = mnist.data.astype(float).to_numpy() / 255.0
        self.y = mnist.target.astype(int).to_numpy()

        if num_points < 70000:
            selectedIndices = np.random.choice(np.arange(self.X.shape[0]), num_points)
            self.X = self.X[selectedIndices]
            self.y = self.y[selectedIndices]

class FashionMNISTDataset(AbstractDataset):
    def __init__(self, num_points: int = 70000, data_home="datasets/fashion_mnist"):
        """
        Create a new Fashion MNIST Dataset.

        num_points determines how many images to keep (chosen randomly)
        num_features is always 784 (28*28).
        """
        super().__init__(num_points, 28*28)
        mnist = fetch_openml('Fashion-MNIST', data_home=data_home)
        self.X = mnist.data.astype(float).to_numpy() / 255.0
        self.y = mnist.target.astype(int).to_numpy()

        if num_points < 70000:
            selectedIndices = np.random.choice(np.arange(self.X.shape[0]), num_points)
            self.X = self.X[selectedIndices]
            self.y = self.y[selectedIndices]
