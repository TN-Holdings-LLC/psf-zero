# psf_compile.py
from qiskit import transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes.synthesis import plugin
import numpy as np
import psf_synthesis 

# ======================================================================
# [Monkey Patch] Forcible Registration into Qiskit Deep Memory
# ======================================================================
_original_init = plugin.UnitarySynthesisPluginManager.__init__

def _hacked_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    class FakeExtension:
        def __init__(self):
            self.name = 'psf_zero'
            self.plugin = psf_synthesis.SU4GeodesicPSFUnitarySynthesis
            self.obj = psf_synthesis.SU4GeodesicPSFUnitarySynthesis()
    fake_ext = FakeExtension()
    self.ext_plugins.extensions.append(fake_ext)
    if hasattr(self.ext_plugins, '_extensions_by_name'):
        if self.ext_plugins._extensions_by_name is None:
            self.ext_plugins._extensions_by_name = {}
        self.ext_plugins._extensions_by_name['psf_zero'] = fake_ext

plugin.UnitarySynthesisPluginManager.__init__ = _hacked_init


def compile(circuit, **kwargs):
    """
    [Latest Confirmed Version] Merges the entire circuit into a single unitary U,
    perfectly restores its unitariness, and passes it directly to PSF-Zero 
    for a one-shot deterministic compression event.
    """
    # 1. Mathematically contract the entire circuit into a single 4x4 matrix
    U = Operator(circuit).data

    # 2. Forcibly restore unitariness (broken by numerical errors) using SVD
    u, _, vh = np.linalg.svd(U)
    U_pure = u @ vh

    # 3. Adjust the global phase to ideal SU(4) (det=1) to eliminate numerical collapse at its root
    det = np.linalg.det(U_pure)
    U_su4 = U_pure / (det ** 0.25)
    
    # 4. Pass to the PSF Synthesizer
    synth = psf_synthesis.SU4GeodesicPSFSynthesizer(psf_synthesis.GeodesicPSFHyper())
    optimized_circ = synth.synthesize(U_su4)
    
    return optimized_circ
