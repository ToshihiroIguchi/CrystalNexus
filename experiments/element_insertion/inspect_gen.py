from pymatgen.analysis.defects.generators import InterstitialGenerator
import inspect

print("--- InterstitialGenerator ---")
print("Init signature:", inspect.signature(InterstitialGenerator.__init__))
print("generate signature:", inspect.signature(InterstitialGenerator.generate))
print("get_defects signature:", inspect.signature(InterstitialGenerator.get_defects))
