from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    sample_rate: int = 160
    n_channels: int = 64
    patch: int = 8
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    context_patches: int = 256
    n_classes: int = 5          # rest / left / right / fists / feet
    data_root: str = r"D:\neurodecode-rt-data"

    @property
    def patch_interval_s(self) -> float:
        return self.patch / self.sample_rate
