"""
Debug script for testing CIF file analysis functionality
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import analyze_cif_file
import asyncio

async def test_cif_analysis():
    """Test CIF analysis with sample files"""
    sample_dir = Path("sample_cif")
    
    if not sample_dir.exists():
        print("Sample CIF directory not found")
        return
    
    cif_files = list(sample_dir.glob("*.cif"))
    
    if not cif_files:
        print("No CIF files found in sample directory")
        return
    
    for cif_file in cif_files:
        print(f"\nAnalyzing: {cif_file.name}")
        print("-" * 40)
        
        try:
            result = await analyze_cif_file(cif_file)
            
            print(f"Formula: {result['formula']}")
            print(f"Space Group: {result['space_group']} (#{result['space_group_number']})")
            print(f"Crystal System: {result['crystal_system']}")
            print(f"Lattice Parameters:")
            print(f"  a = {result['lattice_parameters']['a']:.3f} Å")
            print(f"  b = {result['lattice_parameters']['b']:.3f} Å")
            print(f"  c = {result['lattice_parameters']['c']:.3f} Å")
            print(f"  α = {result['lattice_parameters']['alpha']:.2f}°")
            print(f"  β = {result['lattice_parameters']['beta']:.2f}°")
            print(f"  γ = {result['lattice_parameters']['gamma']:.2f}°")
            print(f"Volume: {result['volume']:.2f} Ų")
            print(f"Number of sites: {result['num_sites']}")
            
        except Exception as e:
            print(f"Error analyzing {cif_file.name}: {e}")

if __name__ == "__main__":
    asyncio.run(test_cif_analysis())