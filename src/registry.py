from typing import Optional
from models.stackdili_fixed.model import Model
from models.stackdili_fixed.model_v6 import ModelV6


def _load_ft(version: str):
    """요청된 FT 버전만 import."""
    if version == "f0":
        from models.stackdili_fixed.ft.ft_v0 import FTv0
        return FTv0
    if version == "f4.5":
        from models.stackdili_fixed.ft.ft_v4_5 import FTv4_5
        return FTv4_5
    raise KeyError(f"FT 버전 '{version}'이 존재하지 않습니다. 가능한 버전: {list(FT_REGISTRY)}")


def _load_stacking(version: str):
    """요청된 Stacking 버전만 import."""
    if version == "s0":
        from models.stackdili_fixed.stacking.stacking_v0 import StackingV0
        return StackingV0
    if version == "s0.5":
        from models.stackdili_fixed.stacking.stacking_v0_5 import StackingV05
        return StackingV05
    if version == "s1":
        from models.stackdili_fixed.stacking.stacking_v1 import StackingV1
        return StackingV1
    if version == "s3":
        from models.stackdili_fixed.stacking.stacking_v3 import StackingV3
        return StackingV3
    raise KeyError(f"Stacking 버전 '{version}'이 존재하지 않습니다. 가능한 버전: {list(STACKING_REGISTRY)}")


# train.py의 choices= 에 사용하기 위한 키 목록 (import 없이 반환)
FT_REGISTRY = {
    "f0":   None,
    "f4.5": None,
    "f6e1": None,   # FTv6 exp1: FP→Q,       ChemBERTa→K,V
    "f6e2": None,   # FTv6 exp2: ChemBERTa→Q, FP→K,V
    "f6e3": None,   # FTv6 exp3: 양방향 Cross-Attention → concat → Linear(32,16)
}

STACKING_REGISTRY = {
    "s0":   None,
    "s0.5": None,
    "s1":   None,
    "s3":   None,
}


def build_model(stacking_version: str, ft_version: Optional[str] = None):
    stacking = _load_stacking(stacking_version)()

    # f6e1/e2/e3 는 ModelV6 전용 파이프라인 사용
    _attn_map = {"f6e1": "fp_query", "f6e2": "chem_query", "f6e3": "bidirect"}
    if ft_version in _attn_map:
        return ModelV6(stacking=stacking, stacking_version=stacking_version,
                       attn_mode=_attn_map[ft_version])

    ft = _load_ft(ft_version)() if ft_version else None
    return Model(stacking=stacking, ft=ft, stacking_version=stacking_version, ft_version=ft_version)
