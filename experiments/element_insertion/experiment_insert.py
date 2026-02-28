import os
import sys
import argparse
from pathlib import Path
from pymatgen.core import Structure, Element, PeriodicSite
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from chgnet.model import CHGNet
import numpy as np

def generate_interstitial_candidates(structure: Structure, insert_element: str) -> list[Structure]:
    """
    pymatgen を用いて格子の空きスペース（Interstitial site）の候補を生成します。
    ここでは、単純な空間的な距離だけでなく、pymatgenのInterstitialGeneratorを
    活用して、物理的により尤もらしい格子間位置の候補を取得します。
    """
    print(f"[{insert_element}] Generating interstitial candididates for the structure...")
    
    # pymatgen の defetcs モジュールにある VoronoiInterstitialGenerator を使用します。
    # これは Voronoi 分割などを用いて、空間的に空いている場所（格子間位置）を
    # 効率的に見つけ出すための標準的な手法です。
    interstitial_gen = VoronoiInterstitialGenerator()
    
    # 候補となる格子間位置（Interstitial）のサイト情報を取得
    # insert_species引数に挿入する元素を指定します。
    candidates_info = interstitial_gen.get_defects(structure, insert_species=[insert_element])
    
    candidate_structures = []
    
    # それぞれの候補地に対して、指定された元素を挿入した Structure オブジェクトを作成
    for i, defect in enumerate(candidates_info):
        # 候補地の分数座標を取得
        frac_coords = defect.site.frac_coords
        
        # 新しい Structure を作成（元の構造をコピー）
        new_structure = structure.copy()
        
        # 候補地に指定した元素を追加
        new_structure.append(insert_element, frac_coords)
        candidate_structures.append(new_structure)
        
    print(f"Found {len(candidate_structures)} geometrically valid interstitial candidate positions.")
    return candidate_structures


def evaluate_candidates(candidate_structures: list[Structure], model_path: str = None) -> tuple[Structure, float]:
    """
    CHGNetを用いて各候補構造のエネルギーを計算し、
    最もエネルギーが低い（= 最も安定した）構造を特定します。
    """
    print("Loading CHGNet model...")
    # CHGNetのロード
    # 軽量な "CHGNet.pretrained" をデフォルトで利用
    chgnet = CHGNet.load()
    
    print("Evaluating energy for each candidate...")
    best_structure = None
    min_energy = float('inf')
    
    # 各候補構造のエネルギーを計算
    for i, struct in enumerate(candidate_structures):
        # 進行状況の表示
        sys.stdout.write(f"\rAssessing candidate {i+1}/{len(candidate_structures)}...")
        sys.stdout.flush()
        
        # CHGNetによる Static (エネルギー) 計算
        prediction = chgnet.predict_structure(struct)
        energy_per_atom = prediction['e']  # eV / atom
        total_energy = energy_per_atom * len(struct) # 比較のため total energy を計算
        
        # 最もエネルギーが低い構造を更新
        if total_energy < min_energy:
            min_energy = total_energy
            best_structure = struct
            
    print(f"\nEvaluation complete. Lowest total energy: {min_energy:.4f} eV")
    
    # 挿入された元素の座標を取得（一番最後に追加されているはず）
    inserted_site = best_structure[-1]
    print(f"Optimal insertion site (fractional): {inserted_site.frac_coords}")
    
    return best_structure, min_energy


def main():
    parser = argparse.ArgumentParser(description="Insert an element into a crystal structure using pymatgen and CHGNet.")
    parser.add_argument("cif_file", help="Path to the input CIF file")
    parser.add_argument("element", help="Element symbol to insert (e.g., Li, H)")
    parser.add_argument("--out", "-o", default="best_inserted.cif", help="Output CIF file path")
    args = parser.parse_args()
    
    input_path = Path(args.cif_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)
        
    print(f"Reading target structure from: {input_path}")
    base_structure = Structure.from_file(input_path)
    print(f"Initial structure info: {base_structure.composition.reduced_formula} ({len(base_structure)} atoms)")
    
    try:
        # Step 1: pymatgenを用いて格子間位置候補を生成
        candidate_structures = generate_interstitial_candidates(base_structure, args.element)
        
        if not candidate_structures:
            print("Could not find any suitable interstitial sites.")
            sys.exit(1)
            
        # Step 2: CHGNetを用いて最も安定な挿入位置を評価・探索
        best_structure, best_energy = evaluate_candidates(candidate_structures)
        
        # Step 3: 結果を保存
        output_path = Path(args.out)
        best_structure.to(filename=str(output_path))
        print(f"Saved optimal structure to: {output_path}")
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
