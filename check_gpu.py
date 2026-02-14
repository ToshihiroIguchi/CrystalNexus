
import torch
import time
import platform

print(f"Platform: {platform.system()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")

try:
    from chgnet.model import CHGNet
    from pymatgen.core import Structure, Lattice

    # Create dummy structure (BaTiO3)
    structure = Structure(
        Lattice.cubic(4.0),
        ["Ba", "Ti", "O", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    )
    # Make supercell 2x2x2 (40 atoms)
    structure.make_supercell([2, 2, 2])
    print(f"Structure size: {len(structure)} atoms")

    # Test CPU
    print("\n--- Testing CPU ---")
    start_time = time.time()
    model_cpu = CHGNet.load(use_device="cpu", verbose=False)
    load_time = time.time() - start_time
    print(f"Model load time (CPU): {load_time:.4f}s")
    
    start_time = time.time()
    model_cpu.predict_structure(structure)
    pred_time = time.time() - start_time
    print(f"Prediction time (CPU): {pred_time:.4f}s")

    # Test GPU if available
    if torch.cuda.is_available():
        print("\n--- Testing GPU ---")
        start_time = time.time()
        model_gpu = CHGNet.load(use_device="cuda", verbose=False)
        load_time = time.time() - start_time
        print(f"Model load time (GPU): {load_time:.4f}s")
        
        start_time = time.time()
        model_gpu.predict_structure(structure)
        pred_time = time.time() - start_time
        print(f"Prediction time (GPU): {pred_time:.4f}s")
    
except ImportError:
    print("CHGNet or pymatgen not installed")
except Exception as e:
    print(f"An error occurred: {e}")
