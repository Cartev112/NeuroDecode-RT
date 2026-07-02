import torch
import torch.nn as nn


def quantize_int8(model):
    """Post-training dynamic int8 quantization of Linear layers (CPU inference).

    Dynamic quantization keeps activations in fp and quantizes Linear weights to
    int8, quantizing activations on-the-fly per forward. This is the pragmatic,
    no-calibration option; it runs on CPU only.
    """
    model = model.to("cpu").eval()
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
