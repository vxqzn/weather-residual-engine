import os
import json
import subprocess
import joblib
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime, timezone

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")

class ModelMetadata(BaseModel):
    model_id: str
    created_at: str
    git_commit: str
    train_mae: float
    holdout_baseline_mae: float
    holdout_model_mae: float
    is_promoted: bool
    promotion_reason: str    

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"

def atomic_save(model, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    joblib.dump(model, temp_path)
    os.replace(temp_path, target_path)
    
def atomic_save_json(data: dict, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    os.replace(temp_path, target_path)

def load_ledger(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> dict:
    ledger_path = artifacts_dir / "ledger.json"
    if not ledger_path.exists():
        return {"active_model": None, "history": []}
    with open(ledger_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_candidate(model, metadata: ModelMetadata, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> None:
    model_path = artifacts_dir / f"{metadata.model_id}.joblib"
    metadata_path = artifacts_dir / f"{metadata.model_id}.json"
    ledger = load_ledger(artifacts_dir)
    
    atomic_save(model, model_path)
    ledger["history"].append(metadata.model_dump())
    atomic_save_json(metadata.model_dump(), metadata_path)
    
    return atomic_save_json(ledger, artifacts_dir / "ledger.json")

def promote_to_champion(model_id: str, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> None:
    candidate_path = artifacts_dir / f"{model_id}.joblib"
    
    if not candidate_path.exists():
        raise FileNotFoundError(f"model {model_id} does not exist.")
    
    model = joblib.load(candidate_path)
    atomic_save(model, artifacts_dir / "champion.joblib")
    
    ledger = load_ledger(artifacts_dir)
    ledger["active_model"] = model_id
    atomic_save_json(ledger, artifacts_dir / "ledger.json")

    return None

def load_champion(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR):
    champion_path = artifacts_dir / "champion.joblib"
    if not champion_path.exists():
        return (None, None)
    
    model = joblib.load(champion_path)
    ledger = load_ledger(artifacts_dir)
    
    active_model_id = ledger.get("active_model")
    entry = next((item for item in ledger["history"] if item["model_id"] == active_model_id), None)
    
    metadata = ModelMetadata(**entry) if entry is not None else None
    
    return (model, metadata)

if __name__ == "__main__":
    # dummy
    model = {"dummy": 1}
    metadata = ModelMetadata(
        model_id="model_dummy",
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        train_mae=0.1,
        holdout_baseline_mae=0.2,
        holdout_model_mae=0.15,
        is_promoted=False,
        promotion_reason=""
    )
    
    save_candidate(model, metadata)
    promote_to_champion("model_dummy")
    
    champion_model, champion_metadata = load_champion()
    
    assert champion_model is not None, "Champion model should not be None"
    assert champion_metadata is not None, "Champion metadata should not be None"