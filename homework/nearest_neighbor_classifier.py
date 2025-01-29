import torch

class NearestNeighborClassifier:
    def __init__(self, x: list[list[float]], y: list[float]):
        self.data, self.label = self.make_data(x, y)
        self.data_mean, self.data_std = self.compute_data_statistics(self.data)
        self.data_normalized = self.input_normalization(self.data)

    @classmethod
    def make_data(cls, x: list[list[float]], y: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor

    @classmethod
    def compute_data_statistics(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.mean(x, dim=0).unsqueeze(0)
        std = torch.std(x, dim=0).unsqueeze(0)
        return mean, std

    def input_normalization(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.data_mean) / self.data_std

    def get_nearest_neighbor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_normalization(x)
        idx = torch.argmin(torch.norm(self.data_normalized - x, dim=1))
        return self.data[idx], self.label[idx]

    def get_k_nearest_neighbor(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_normalization(x)
        idx = torch.argsort(torch.norm(self.data_normalized - x, dim=1))[:k]
        return self.data[idx], self.label[idx]

    def knn_regression(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return self.get_k_nearest_neighbor(x, k)[1].mean()
