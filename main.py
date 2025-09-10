import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

from dataset import AbstractDataset, \
    UniformRandomDataset, \
    PrototypeDataset, \
    MNISTDataset, \
    FashionMNISTDataset

from FSCL import FSCL

# --------------------------------------------------------------------------------

# NUM_POINTS = 1000
# NUM_FEATURES = 2
# NUM_PROTOTYPES = 4
# INSTANCE_NOISE = 0.2
# dataset = PrototypeDataset(NUM_PROTOTYPES,INSTANCE_NOISE,NUM_POINTS,NUM_FEATURES)

NUM_PROTOTYPES = 10
NUM_POINTS = 1000
dataset = FashionMNISTDataset(NUM_POINTS)
NUM_FEATURES = dataset.num_features

clustering = FSCL(NUM_PROTOTYPES, NUM_FEATURES)
clustering.fit(dataset.X)
predicted_labels = clustering.predict(dataset.X)

silhouette = silhouette_score(dataset.X, predicted_labels)
nmi = normalized_mutual_info_score(dataset.y, predicted_labels)
print(f"Silhouette Score; {silhouette}\nNormalized Mutual Information: {nmi}")
