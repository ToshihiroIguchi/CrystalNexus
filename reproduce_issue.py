import os
import sys
import numpy as np
# Make sure we can import chgnet
try:
    from chgnet.model import CHGNet, StructOptimizer
    from pymatgen.core import Structure, Lattice
except ImportError:
    print("CHGNet or pymatgen not installed.")
    sys.exit(1)

def create_simple_structure():
    # Simple cubic BaTiO3
    lattice = Lattice.from_parameters(a=4.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=90)
    structure = Structure(lattice, ["Ba", "Ti", "O", "O", "O"],
                         [[0, 0, 0], [0.5, 0.5, 0.5], 
                          [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return structure

async def reproduce():
    print("Loading CHGNet...")
    model = CHGNet.load(use_device="cpu", verbose=False)
    relaxer = StructOptimizer(model=model, use_device="cpu", optimizer_class="FIRE")
    
    # Test 1: steps=0 content
    print("\n--- Test 1: steps=0 content ---")
    structure = create_simple_structure()
    result = relaxer.relax(structure, fmax=0.1, steps=0, verbose=False)
    traj = result['trajectory']
    print(f"steps=0 length: {len(traj)}")
    e0 = traj.energies[0]
    e1 = traj.energies[1]
    print(f"Energy 0: {e0}")
    print(f"Energy 1: {e1}")
    print(f"Energies equal: {e0 == e1}")

    # Test 2: Early stopping
    # Use very loose fmax so it converges immediately
    print("\n--- Test 2: Early stopping (fmax=10.0) ---")
    structure = create_simple_structure()
    # Initial forces should be smallish, but let's see. BaTiO3 unrelaxed might have forces.
    result = relaxer.relax(structure, fmax=100.0, steps=5, verbose=False)
    traj = result['trajectory']
    print(f"Request steps=5, fmax=100.0 (loose)")
    print(f"Result length: {len(traj)}")
    
    # Test 3: steps=2 with tight fmax (force max steps)
    print("\n--- Test 3: steps=2 with tight fmax ---")
    structure = create_simple_structure()
    result = relaxer.relax(structure, fmax=0.0001, steps=2, verbose=False)
    traj = result['trajectory']
    print(f"Request steps=2, fmax=0.0001")
    print(f"Result length: {len(traj)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(reproduce())
