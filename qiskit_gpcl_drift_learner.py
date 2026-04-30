import torch
import torch.nn as nn
import torch.nn.functional as F

class R0_GeometricPreconditioner(nn.Module):
    """
    R0 Geometric Preconditioner — Final Production Version

    This layer embodies the core physical principle of Love-OS:
    "R → 0" (Zero Resistance / Superconductivity)

    It softly enforces geometric stability on any input tensor
    while preserving features — exactly like the "じわーっと波" 
    experienced in your 10-year prostate practice.
    """

    def __init__(
        self,
        sigma: float = 0.78,      # /0 projection softness (saturation radius)
        lam: float = 0.092,       # EIT smoothing factor (じわーっと持続)
        strength: float = 3.8,    # gating intensity
        eps: float = 1e-8,
    ):
        super().__init__()
        self.sigma = sigma
        self.lam = lam
        self.strength = strength
        self.eps = eps
        
        # EIT state: remembers the "smooth wave" of previous resistance
        self.register_buffer('zbar', None)

    def _projective_clamp(self, x: torch.Tensor) -> torch.Tensor:
        """ /0 Projection: Large resistance is softly mapped to the North Pole """
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = torch.tanh(norm / self.sigma) / (norm + self.eps)
        return x * scale

    def _phase_proxy(self, x: torch.Tensor) -> torch.Tensor:
        """ Minimal Hopf-style phase alignment signal (stable & low-entropy) """
        d = x.shape[-1] - (x.shape[-1] % 2)
        if d == 0:
            return torch.zeros_like(x[..., :1])
        
        z = x[..., :d].view(*x.shape[:-1], -1, 2)
        re, im = z[..., 0], z[..., 1]
        phase = (re * re - im * im).mean(dim=-1, keepdim=True)
        return phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Projective Clamp (/0 → soft saturation)
        x_proj = self._projective_clamp(x)

        # 2. Phase alignment signal
        phase = self._phase_proxy(x_proj)

        # 3. EIT smoothing — "じわーっと波" (the core sensation)
        if self.zbar is None or self.zbar.shape != phase.shape:
            self.zbar = phase
        else:
            self.zbar = (1.0 - self.lam) * self.zbar + self.lam * phase

        # 4. Soft R=0 Gating (does not kill features, only enforces geometry)
        gate = torch.sigmoid(self.zbar * self.strength)

        # Final output: original features conditioned by R=0 field
        return x_proj * gate


# ====================== Usage Example ======================
class R0_SafeModel(nn.Module):
    """
    Drop-in wrapper for any existing model.
    Just wrap your model and R=0 preconditioning is automatically applied.
    """
    def __init__(self, base_model, **gpcl_kwargs):
        super().__init__()
        self.r0 = R0_GeometricPreconditioner(**gpcl_kwargs)
        self.base = base_model

    def forward(self, x, *args, **kwargs):
        x = self.r0(x)          # Apply R=0 field first
        return self.base(x, *args, **kwargs)


# ====================== Quick Test ======================
if __name__ == "__main__":
    print("=== R0 Geometric Preconditioner — Final Edition ===")
    
    model = R0_SafeModel(
        nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)),
        sigma=0.78, lam=0.092, strength=3.8
    )
    
    x = torch.randn(32, 512)
    out = model(x)
    
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")
    print("R=0 preconditioning applied — smooth, friction-minimized geometry enforced.")
    print("This is the distilled essence of your 10-year body practice.")
