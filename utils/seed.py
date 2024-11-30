def seed_everything(seed: int, deterministic: bool = True, benchmark: bool = True):
    """
    Set the seed for generating random numbers to ensure reproducibility.

    This function sets the seed for Python's `random` module, NumPy, and PyTorch.
    It also configures PyTorch's CUDA backend to be deterministic or to use
    benchmark mode for performance.

    Args:
        seed (int): The seed value to use for random number generation.
        deterministic (bool, optional): If True, sets PyTorch to use deterministic algorithms. Defaults to True.
        benchmark (bool, optional): If True, enables the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware. Defaults to True.
    """
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
