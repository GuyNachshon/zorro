#!/usr/bin/env python3
"""Clear GPU memory and show usage."""

import torch
import gc

def clear_gpu_memory():
    """Clear GPU memory cache."""
    print("🧹 Clearing GPU memory...")
    
    # Show current usage
    if torch.cuda.is_available():
        print(f"  Current allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Current reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        print(f"  After clearing:")
        print(f"    Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"    Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("  No GPU available")

if __name__ == "__main__":
    clear_gpu_memory()