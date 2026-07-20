from .scale_shift import ScaleShiftMLP, ScaleShiftClassifier

__all__ = ["BSMambaEncoderDecoder", "DGRDLModel", "ScaleShiftMLP", "ScaleShiftClassifier"]


def __getattr__(name: str):
    # Lazy-import mamba-dependent modules so baselines can load without mamba-ssm.
    if name in {"BSMambaEncoderDecoder", "DGRDLModel"}:
        from .bsmamba import BSMambaEncoderDecoder, DGRDLModel

        return {"BSMambaEncoderDecoder": BSMambaEncoderDecoder, "DGRDLModel": DGRDLModel}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
