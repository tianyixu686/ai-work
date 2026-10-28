"""Fast checks that do not require PyTorch or a dataset download."""
from pathlib import Path
from expert_system import recognize

root = Path(__file__).resolve().parents[1]
for name in ("cat", "dog", "deer", "horse"):
    result = recognize(root / "examples" / f"{name}.png")
    assert result["status"] in {"recognized", "unknown"}
    assert len(result["statistics"]) == 7
    print(name, "->", result["prediction"], result["scores"])
print("smoke test passed")

