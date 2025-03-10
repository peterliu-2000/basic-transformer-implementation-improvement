import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    # Get the current memory usage
    current_memory_allocated = torch.mps.current_allocated_memory()
    max_memory_allocated = torch.mps.driver_allocated_memory()

    print(f"Current memory allocated on MPS: {current_memory_allocated / 1024 ** 2:.2f} MB")
    print(f"Max memory allocated on MPS: {max_memory_allocated / 1024 ** 2:.2f} MB")
else:
    print("MPS is not available.")

torch.mps.empty_cache()