#!/usr/bin/env python3
"""
chgnet_relax_batio3_verbose.py

BaTiO3.cif を読み込んで、CHGNet（0.3.0, CPU）で予測 → 緩和 → 再予測 → 出力結果をコンソールに全出力するスクリプト。

特長:
- エラーハンドリングで例外は隠さず表示・再スロー
- TrajectoryObserver を使ってトラジェクトリを安全に保存
- 結果は JSON にも書き出すが、同時に全結果を標準出力にも表示
"""

import sys
import traceback
from pathlib import Path
import json

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from chgnet.model.model import CHGNet
from chgnet.model import StructOptimizer
from chgnet.model.dynamics import TrajectoryObserver

def safe_get_prediction(pred):
    out = {}
    for k in ("energy", "e", "energy_per_atom", "e_per_atom"):
        if k in pred:
            out["energy_eV_per_atom"] = float(getattr(pred[k], "item", lambda: pred[k])())
            break
    for k in ("forces", "f", "force"):
        if k in pred:
            arr = pred[k]
            out["forces_eV_per_A"] = [list(a) for a in arr.tolist()]
            break
    for k in ("stress", "s", "virial"):
        if k in pred:
            arr = pred[k]
            out["stress_GPa"] = [list(a) for a in arr.tolist()]
            break
    for k in ("magmom", "m", "magmoms"):
        if k in pred:
            arr = pred[k]
            out["magmoms_muB"] = [float(x) for x in arr.tolist()]
            break
    # 他の小さなフィールドも自動追加
    for k, v in pred.items():
        if k in ("energy", "e", "forces", "f", "stress", "s", "magmom", "m"):
            continue
        try:
            out[k] = v.tolist() if hasattr(v, "tolist") else v
        except:
            continue
    return out

def main(cif_path, outdir="chgnet_out"):
    cif = Path(cif_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not cif.exists():
        raise FileNotFoundError(f"CIF が見つかりません: {cif}")

    print(f"--- 入力構造を読み込み中: {cif} ---")
    structure = Structure.from_file(str(cif))

    print("--- CHGNet モデル（0.3.0, CPU）をロード中 ---")
    chgnet = CHGNet.load(model_name="0.3.0", use_device="cpu", verbose=True)
    print(f"モデルパラメータ数: {chgnet.n_params}, バージョン: {chgnet.version}")

    print("\n--- 入力構造を予測中（エネルギー、力、応力、磁気モーメント、可能であればサイトエネルギー等）---")
    try:
        pred_in = chgnet.predict_structure(structure,
                                           return_site_energies=True,
                                           return_atom_feas=True,
                                           return_crystal_feas=True)
    except TypeError:
        pred_in = chgnet.predict_structure(structure)
    except Exception:
        print("入力構造の予測中にエラー発生:")
        traceback.print_exc()
        raise
    in_props = safe_get_prediction(pred_in)
    print("===== 入力構造の予測結果 =====")
    print(json.dumps(in_props, indent=2, ensure_ascii=False))
    (outdir / "input_prediction.json").write_text(json.dumps(in_props, indent=2))

    print("\n--- 構造緩和開始（FIRE, CPU）---")
    try:
        relaxer = StructOptimizer(model=chgnet, use_device="cpu", optimizer_class="FIRE")
        result = relaxer.relax(structure, fmax=0.1, loginterval=1, verbose=True, assign_magmoms=True)
    except Exception:
        print("構造緩和中にエラー発生:")
        traceback.print_exc()
        raise

    final = result.get("final_structure")
    traj = result.get("trajectory")
    if final is None:
        raise RuntimeError("final_structure が返されませんでした。Result keys: " + str(result.keys()))

    relaxed_cif = outdir / "BaTiO3_relaxed_by_CHGNet.cif"
    print(f"\n--- 緩和後構造を CIF に保存: {relaxed_cif}")
    try:
        CifWriter(final).write_file(str(relaxed_cif))
    except:
        final.to(filename=str(relaxed_cif))

    if traj is not None and isinstance(traj, TrajectoryObserver):
        try:
            traj_path = outdir / "relax_trajectory.traj"
            traj.save(str(traj_path))
            print(f"トラジェクトリを保存: {traj_path}")
        except Exception:
            print("トラジェクトリの保存に失敗:")
            traceback.print_exc()
    else:
        print("トラジェクトリ情報がありません。保存をスキップします。")

    print("\n--- 緩和構造物性を予測中 ---")
    try:
        pred_rel = chgnet.predict_structure(final,
                                            return_site_energies=True,
                                            return_atom_feas=True,
                                            return_crystal_feas=True)
    except TypeError:
        pred_rel = chgnet.predict_structure(final)
    except Exception:
        print("緩和後構造の予測中にエラー:")
        traceback.print_exc()
        raise

    rel_props = safe_get_prediction(pred_rel)
    print("===== 緩和後構造の予測結果 =====")
    print(json.dumps(rel_props, indent=2, ensure_ascii=False))
    (outdir / "relaxed_prediction.json").write_text(json.dumps(rel_props, indent=2))

    summary = {
        "input_energy_eV_per_atom": in_props.get("energy_eV_per_atom"),
        "relaxed_energy_eV_per_atom": rel_props.get("energy_eV_per_atom")
    }
    print("\n===== エネルギー比較 =====")
    print(f"入力： {summary['input_energy_eV_per_atom']} eV/atom")
    print(f"緩和後： {summary['relaxed_energy_eV_per_atom']} eV/atom")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nすべての結果はディレクトリ {outdir.resolve()} に出力済み。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python chgnet_relax_batio3_verbose.py BaTiO3.cif [出力先ディレクトリ]")
        sys.exit(1)
    cif = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) >= 3 else "chgnet_out"
    main(cif, outdir)
