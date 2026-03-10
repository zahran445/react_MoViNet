"""
Cross-platform device detection utility
Supports: CUDA (NVIDIA), MPS (Apple Silicon), CPU
"""
import torch

def get_device(verbose=True):
    """
    Get the best available device for PyTorch.
    Priority: CUDA > MPS > CPU
    
    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        if verbose:
            print(f"[DEVICE] Using CUDA: {device_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print(f"[DEVICE] Using Apple Metal (MPS) - GPU acceleration enabled")
    else:
        device = torch.device("cpu")
        if verbose:
            print(f"[DEVICE] Using CPU (slow - consider upgrading hardware)")
    
    return device

def get_device_info():
    """Get detailed device information."""
    info = {
        'device_type': 'cpu',
        'device_name': 'CPU',
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info['device_type'] = 'cuda'
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device_type'] = 'mps'
        info['device_name'] = 'Apple Silicon GPU'
    
    return info

if __name__ == '__main__':
    print("=" * 60)
    print("PyTorch Device Detection")
    print("=" * 60)
    
    device = get_device(verbose=True)
    info = get_device_info()
    
    print(f"\nDevice Type: {info['device_type']}")
    print(f"Device Name: {info['device_name']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"MPS Available: {info['mps_available']}")
    
    if 'cuda_version' in info:
        print(f"CUDA Version: {info['cuda_version']}")
    
    print(f"\nPyTorch Version: {torch.__version__}")
    print("=" * 60)
