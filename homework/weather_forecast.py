import torch
from typing import Tuple

class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        min_per_day, _ = self.data.min(dim=1)
        max_per_day, _ = self.data.max(dim=1)
        return min_per_day, max_per_day

    def find_the_largest_drop(self) -> torch.Tensor:
        return self.data.mean(dim=1).diff().min()

    def find_the_most_extreme_day(self) -> torch.Tensor:
        each_day_avg = self.data.mean(dim=1, keepdim=True)
        max_deviation_indices = (self.data - each_day_avg).abs().max(dim=1).indices
        max_deviation_values = self.data[torch.arange(self.data.size(0)), max_deviation_indices]
        return max_deviation_values

    def max_last_k_days(self, k: int) -> torch.Tensor:
        return self.data[-k:].max(dim=1).values

    def predict_temperature(self, k: int) -> torch.Tensor:
        return self.data[-k:].mean()

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        return torch.argmin((self.data - t).abs().sum(dim=1))
