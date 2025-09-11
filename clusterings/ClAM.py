import numpy as np
from sklearn.base import ClusterMixin
import torch
from tqdm import tqdm

class ClAMClustering(ClusterMixin):
    """
    Basic Clustering with Associative Memories
    """

    def __init__(self, 
                    num_prototypes: int, 
                    num_features: int,
                    torch_device: str = "cuda",
                    beta: float = 1,
                    time_constant: float = 1,
                    num_iterations: int = 8,
                    optimizer_kwargs: dict = {},
                ) -> None:
        super().__init__()

        self.num_prototypes = num_prototypes
        self.n_clusters = num_prototypes
        self.num_features = num_features
        self.torch_device = torch_device
        self.beta = beta
        self.time_constant = time_constant
        self.num_iterations = num_iterations

        self.prototypes = torch.normal(torch.zeros(num_prototypes, num_features), 0.01).to(self.torch_device).requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.prototypes], **optimizer_kwargs) # type: ignore

    def __str__(self) -> str:
        return f"ClAM Clustering (num_prototypes={self.num_prototypes})"

    def fit(self, 
            X: torch.Tensor, 
            num_epochs: int = 1000, 
            batch_size: int = 512, 
            mask_bernoulli_parameter: float = 0.8,
            disable_progress_bar: bool = False
        ) -> list[float]:
        """
        Fit the dataset defined by X.
        X has shape (num_points, num_features).

        Returns the (epoch) loss history over training
        """
        
        loss_history = []

        num_batches = int(np.ceil(X.shape[0] / batch_size))
        for epoch_index in (epoch_progress_bar := tqdm(range(num_epochs), desc="Epoch", disable=disable_progress_bar, leave=False)):
            epoch_total_loss = torch.zeros(1).to(self.torch_device)
            epoch_data_random_indices = torch.randperm(X.shape[0])
            for batch_index in range(num_batches):
                batch_loss = 0

                batch_data_random_indices = epoch_data_random_indices[batch_index*batch_size:(batch_index+1)*batch_size]
                # Shape (batch_size, num_features)
                batch_data = X[batch_data_random_indices].to(self.torch_device)
                del batch_data_random_indices

                # Shape (batch_size, num_features)
                batch_masks = torch.bernoulli(mask_bernoulli_parameter * torch.ones_like(batch_data))
                batch_masks_compliment = 1 - batch_masks
                probes = batch_masks * batch_data.clone()
                
                for iteration_index in range(self.num_iterations):
                    # Algorithm 1, line 10
                    # Shape (num_prototypes, batch_size, num_features)
                    delta_mu = self.prototypes[:, None, :] - probes[None, :, :]
                    # avg_distance = delta_mu.square().sum(dim=2).min(dim=0).values.mean()
                    
                    # Algorithm 1, line 11 (softmax / sigma)
                    # Shape (num_prototypes, batch_size)
                    softmax_result = torch.softmax(-self.beta * delta_mu.square().sum(dim=2), dim=0)
                    
                    # Algorithm 1, line 11 (inside sum over mu)
                    # Shape (num_prototypes, batch_size, num_features)
                    weighted_update = (delta_mu * softmax_result[:, :, None]).sum(dim=0)
                    
                    # Algorithm 1, line 11 (complete)
                    # Shape (batch_size, num_features)
                    delta = batch_masks_compliment * 1/self.time_constant * weighted_update
                    
                    # Algorithm 1, line 12
                    probes = probes + delta
                # Shape (batch_size,)
                squared_distance = (batch_masks_compliment * (probes-batch_data)).square().sum(dim=1)
                batch_loss += squared_distance.mean(dim=0)
                with torch.no_grad():
                    epoch_total_loss += batch_loss

                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_history.append(epoch_total_loss.item())
            epoch_progress_bar.set_postfix_str(f"Loss {epoch_total_loss.item():.4e}")
        return loss_history

    @torch.no_grad()
    def predict(self, 
                X: torch.Tensor,
                batch_size: int = 512, 
            ) -> torch.Tensor:
        """
        Predict the classes of a dataset X.
        X has shape (num_points, num_features).
        """

        predictions = []
        num_batches = int(np.ceil(X.shape[0] / batch_size))
        for batch_index in range(num_batches):
            batch_data = X[batch_index*batch_size:(batch_index+1)*batch_size].to(self.torch_device)
            probes = batch_data.clone().detach()
            
            for iteration_index in range(self.num_iterations):
                # Algorithm 1, line 21
                # Shape (num_prototypes, batch_size, num_features)
                delta_mu = self.prototypes[:, None, :] - probes[None, :, :]
                
                # Algorithm 1, line 22 (softmax / sigma)
                # Shape (num_prototypes, batch_size)
                softmax_result = torch.softmax(-self.beta * delta_mu.square().sum(dim=2), dim=0)
                
                # Algorithm 1, line 22 (inside sum over mu)
                # Shape (num_prototypes, batch_size, num_features)
                weighted_update = delta_mu * softmax_result[:, :, None]
                
                # Algorithm 1, line 2 (complete)
                # Shape (batch_size, num_features)
                probes = probes + 1/self.time_constant * weighted_update.sum(dim=0)
            
            # Algorithm 1, line 23
            # Shape (num_prototypes, batch_size)
            squared_distance = (self.prototypes[:, None, :] - probes[None, :, :]).square().sum(dim=2)
            batch_predictions = squared_distance.argmin(dim=0)
            predictions.append(batch_predictions)
        predictions = torch.cat(predictions)
        return predictions

    def fit_predict(self, X: np.ndarray, y = None) -> np.ndarray:
        torch_X = torch.tensor(X) 
        _ = self.fit(torch_X)
        torch_predicted_labels = self.predict(torch_X)
        return torch_predicted_labels.cpu().numpy()
    
class RegularizedClAM(ClAMClustering):
    """
    The same idea as standard ClAM, but include a quadratic penalty term to the loss
    the encourages memory vectors to spread out. This term:
    $ \sum_{\mu \neq \nu} 1 / (\lvert\lvert  \rho_\mu - \rho_\nu \rvert\rvert^2 + \epsilon) $
    goes to zero (technically $1/\epsilon$ as the distance between vectors grows, meaning it will not blow up.
    """

    def __init__(self, 
                    num_prototypes: int, 
                    num_features: int, 
                    regularization_lambda: float,
                    regularization_exponent: float = 2,
                    torch_device: str = "cuda", 
                    beta: float = 1, 
                    time_constant: float = 1, 
                    num_iterations: int = 8,
                    optimizer_kwargs: dict = {}
                ) -> None:
        
        super().__init__(num_prototypes, 
            num_features, 
            torch_device, 
            beta, 
            time_constant, 
            num_iterations, 
            optimizer_kwargs    
        )
        self.regularization_lambda = regularization_lambda
        self.regularization_exponent = regularization_exponent

    def __str__(self) -> str:
        return f"ClAM Clustering with Regularization (num_prototypes={self.num_prototypes})"

    def fit(self, 
            X: torch.Tensor, 
            num_epochs: int = 1000, 
            batch_size: int = 512, 
            mask_bernoulli_parameter: float = 0.8,
            disable_progress_bar: bool = False
        ) -> list[float]:
        """
        Fit the dataset defined by X.
        X has shape (num_points, num_features).

        Returns the (epoch) loss history over training
        """
        
        loss_history = []

        num_batches = int(np.ceil(X.shape[0] / batch_size))
        for epoch_index in (epoch_progress_bar := tqdm(range(num_epochs), desc="Epoch", disable=disable_progress_bar, leave=False)):
            epoch_total_loss = torch.zeros(1).to(self.torch_device)
            epoch_data_random_indices = torch.randperm(X.shape[0])
            for batch_index in range(num_batches):
                batch_loss = 0

                batch_data_random_indices = epoch_data_random_indices[batch_index*batch_size:(batch_index+1)*batch_size]
                # Shape (batch_size, num_features)
                batch_data = X[batch_data_random_indices].to(self.torch_device)
                del batch_data_random_indices

                # Shape (batch_size, num_features)
                batch_masks = torch.bernoulli(mask_bernoulli_parameter * torch.ones_like(batch_data))
                batch_masks_compliment = 1 - batch_masks
                probes = batch_masks * batch_data.clone()
                
                for iteration_index in range(self.num_iterations):
                    # Algorithm 1, line 10
                    # Shape (num_prototypes, batch_size, num_features)
                    delta_mu = self.prototypes[:, None, :] - probes[None, :, :]
                    # avg_distance = delta_mu.square().sum(dim=2).min(dim=0).values.mean()
                    
                    # Algorithm 1, line 11 (softmax / sigma)
                    # Shape (num_prototypes, batch_size)
                    softmax_result = torch.softmax(-self.beta * delta_mu.square().sum(dim=2), dim=0)
                    
                    # Algorithm 1, line 11 (inside sum over mu)
                    # Shape (num_prototypes, batch_size, num_features)
                    weighted_update = (delta_mu * softmax_result[:, :, None]).sum(dim=0)
                    
                    # Algorithm 1, line 11 (complete)
                    # Shape (batch_size, num_features)
                    delta = batch_masks_compliment * 1/self.time_constant * weighted_update
                    
                    # Algorithm 1, line 12
                    probes = probes + delta
                # Shape (batch_size,)
                squared_distance = (batch_masks_compliment * (probes-batch_data)).abs().pow(self.regularization_exponent).sum(dim=1)
                batch_loss += squared_distance.mean(dim=0)

                # Regularization term, encouraging dissimilar vectors
                prototype_similarities = (self.prototypes[:, None, :] - self.prototypes[None, :, :]).square().sum(dim=2)
                similarity_mask = ~torch.eye(self.num_prototypes, dtype=torch.bool, device=self.torch_device)
                batch_loss += self.regularization_lambda * (1/prototype_similarities[similarity_mask]).sum()

                with torch.no_grad():
                    epoch_total_loss += batch_loss

                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_history.append(epoch_total_loss.item())
            epoch_progress_bar.set_postfix_str(f"Loss {epoch_total_loss.item():.4e}")
        return loss_history