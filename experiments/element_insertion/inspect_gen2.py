from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
import inspect

print("--- VoronoiInterstitialGenerator ---")
print("Init signature:", inspect.signature(VoronoiInterstitialGenerator.__init__))
print("generate signature:", inspect.signature(VoronoiInterstitialGenerator.generate))
print("get_defects signature:", inspect.signature(VoronoiInterstitialGenerator.get_defects))
