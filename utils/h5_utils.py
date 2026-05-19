"""
HDF5 file utilities for reading HDF5 data.
Code logic adapted from icl_model to avoid external dependencies.
"""

import h5py
import numpy as np
import torch
from typing import Optional


class H5:
    """HDF5 file reader utility class."""
    
    @classmethod
    def open_file(cls, file, mode: str = 'r') -> tuple:
        """Open an HDF5 file and return file object and close flag."""
        if isinstance(file, str):
            file = h5py.File(file, mode)
            to_close = True
        elif isinstance(file, h5py.File):
            to_close = False
        else:
            raise ValueError(f'Invalid file {file.__class__}')
        return file, to_close
    
    @classmethod
    def read_tensor(
        cls,
        file,
        group: str,
        dtype: torch.dtype = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """Read an HDF5 group as a PyTorch tensor.
        
        Args:
            file: HDF5 file path (str) or file object (h5py.File)
            group: The group/dataset name to read
            dtype: Target torch.dtype (optional)
            device: Target torch.device (optional)
            
        Returns:
            torch.Tensor: The data as a PyTorch tensor
        """
        file, to_close = cls.open_file(file, mode='r')
        data = torch.from_numpy(np.asarray(file[group]))
        if to_close:
            file.close()
        if dtype is not None or device is not None:
            data = data.to(device=device, dtype=dtype)
        return data