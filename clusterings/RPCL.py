import numpy as np
from sklearn.base import ClusterMixin
from tqdm import tqdm

class RPCL(ClusterMixin):
    """
    Rival Penalized Competitive Learning
    Like FSCL, but now move the 2nd best unit *away* from data probes.
    Should prevent parasitic prototypes that stabilize between clusters.
    """
    
    def __init__(self, 
                 num_prototypes: int, 
                 num_features: int
            ) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototypes = np.random.normal(0, 1, size=(self.num_prototypes, num_features))
        self.winning_ratios = np.ones(self.num_prototypes)

    def fit(self, 
            X: np.ndarray, 
            num_epochs: int = 512,
            best_matching_unit_learning_rate: float = 1e-3, 
            rival_matching_unit_learning_rate: float = 1e-5, 
            batch_size = 64,
            disable_progress_bar: bool = False,
        ) -> None:
        self.winning_ratios = np.ones(self.num_prototypes)

        num_batches = int(np.ceil(X.shape[0] / batch_size))
        epoch_data_random_indices = np.arange(X.shape[0])
        for epoch_index in (epoch_progress_bar := tqdm(range(num_epochs), desc="Epoch", disable=disable_progress_bar, leave=False)):
            np.random.shuffle(epoch_data_random_indices)
            for batch_index in range(num_batches):

                batch_data_random_indices = epoch_data_random_indices[batch_index*batch_size:(batch_index+1)*batch_size]
                # Shape (batch_size, num_features)
                batch_data = X[batch_data_random_indices]
                del batch_data_random_indices

                # For each data point in the batch, find the best matching unit

                # Difference between prototypes and each datapoint,
                # Shape (num_prototypes, batch_size, num_features)
                delta_mu = self.prototypes[:, None, :] - batch_data[None, :, :]

                # Squared distance between prototypes and batch data
                # Shape (num_prototypes, batch_size)
                squared_distance: np.ndarray = np.square(delta_mu).sum(axis=2)
                weighted_squared_distance = (self.winning_ratios / np.sum(self.winning_ratios))[:, None] * squared_distance

                # Index of the best matching unit for each datum
                # Shape (batch_size,)
                argpart = np.argpartition(weighted_squared_distance, kth=[0,1], axis=0)
                best_matching_unit_index: np.ndarray = argpart[0, :]
                rival_matching_unit_index: np.ndarray = argpart[1, :]
                del argpart

                # Get an indicator matrix for the best matching unit against each prototype
                # Shape (num_prototypes, batch_size, num_features)
                best_matching_unit_mask = np.zeros_like(delta_mu)
                best_matching_unit_mask[best_matching_unit_index, np.arange(batch_data.shape[0]), :] = 1
                rival_matching_unit_mask = np.zeros_like(delta_mu)
                rival_matching_unit_mask[rival_matching_unit_index, np.arange(batch_data.shape[0]), :] = 1

                # Update the winning ratios for units that just won
                unique_best_matching_unit_index, count_unique_best_matching_unit_index = np.unique_counts(best_matching_unit_index)
                self.winning_ratios[unique_best_matching_unit_index] += count_unique_best_matching_unit_index
                del unique_best_matching_unit_index
                del count_unique_best_matching_unit_index

                # The update to each prototype according to each datum
                # Shape (num_prototypes, num_features)
                prototype_updates = best_matching_unit_learning_rate * (best_matching_unit_mask * delta_mu).sum(axis=1)
                self.prototypes -= prototype_updates

                prototype_updates = rival_matching_unit_learning_rate * (rival_matching_unit_mask * delta_mu).sum(axis=1)
                self.prototypes += prototype_updates
        
    def predict(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        predictions = []
        num_batches = int(np.ceil(X.shape[0] / batch_size))
        for batch_index in range(num_batches):
            batch_data = X[batch_index*batch_size:(batch_index+1)*batch_size]
            delta_mu = self.prototypes[:, None, :] - batch_data[None, :, :]
            squared_distance: np.ndarray = np.square(delta_mu).sum(axis=2)
            best_matching_unit_index: np.ndarray = squared_distance.argmin(axis=0)
            predictions.append(best_matching_unit_index)
        predictions = np.concat(predictions)
        return predictions


    def fit_predict(self, X: np.ndarray, y = None, **fit_args) -> np.ndarray:
        self.fit(X, **fit_args)
        return self.predict(X)