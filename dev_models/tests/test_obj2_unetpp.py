"""UNet++ internal baseline of Table 4.2 (src.models.unetpp_notebook): the recipe the retrain uses."""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.unetpp_notebook import UNetPlusPlus_EfficientNetB5  # noqa: E402


def test_unetpp_outputs_logit_tuple_in_training_and_probabilities_in_eval():
    m = UNetPlusPlus_EfficientNetB5(encoder_weights=None)
    x = torch.rand(1, 3, 64, 64)
    out = m.train()(x)
    assert isinstance(out, tuple) and out[0].shape == (1, 1, 64, 64)
    p = m.eval()(x)
    assert p.min() >= 0 and p.max() <= 1


def test_unetpp_retrain_recipe_one_group_nothing_frozen():
    spec = importlib.util.spec_from_file_location("rt", ROOT / "scripts" / "obj2_retrain_siran.py")
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    m, frozen = rt.build("unetpp", 0.1)
    assert frozen == [] and getattr(m, "stem", None) is None          # the notebook's "stem" names no module
    assert [len(list(m.parameters()))] == rt.ARCH["unetpp"]["groups"]
    assert rt.ARCH["unetpp"]["batch"] == 8


def test_unetpp_retrain_loads_a_wrapped_state_dict(tmp_path, monkeypatch):
    # the phase-1 checkpoint's keys start with "model." (the wrapper); the retrain must not strip them
    m = UNetPlusPlus_EfficientNetB5(encoder_weights=None)
    ck = tmp_path / "sbu.pth"; torch.save({"model_state_dict": m.state_dict()}, ck)
    src = (ROOT / "scripts" / "obj2_retrain_siran.py").read_text()
    assert 'if args.arch != "unetpp"' in src
    sd = torch.load(ck, weights_only=False)["model_state_dict"]
    assert all(k.startswith("model.") for k in sd)
    UNetPlusPlus_EfficientNetB5(encoder_weights=None).load_state_dict(sd, strict=True)
