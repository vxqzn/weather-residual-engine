import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from twre.models.registry import (
    DEFAULT_ARTIFACTS_DIR,
    ModelMetadata,
    get_git_commit,
    load_champion,
    promote_to_champion,
    save_candidate,
)
from twre.models.evaluate import evaluate_model


def evaluate_and_promote(
    candidate_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    margin: float = 0.00,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
) -> tuple[ModelMetadata, bool]:
    training_metrics = evaluate_model(candidate_model, X_train, y_train)
    holdout_metrics = evaluate_model(candidate_model, X_holdout, y_holdout)
    
    candidate_mae = holdout_metrics["model_mae"]
    baseline_mae = holdout_metrics["baseline_mae"]
    
    champion_model, champion_metadata = load_champion(artifacts_dir)
    
    if champion_model is None or champion_metadata is None:
        is_promoted = bool(candidate_mae < baseline_mae)
        if is_promoted:
            reason = f"cold start candidate model promoted as there is currently no champion (cold start)"
        else:
            reason = f"cold start candidate model not promoted, doesn't outperform the baseline"
    else:
        champion_metrics = evaluate_model(champion_model, X_holdout, y_holdout)
        champion_mae = champion_metrics["model_mae"]
        delta = champion_mae - candidate_mae
        is_promoted = bool(candidate_mae < (champion_mae - margin))
        if is_promoted:
            reason = f"candidate model promoted, outperforms current champion by {delta:.4f} degC (margin={margin:.4f} degC)"
        else:
            reason = f"candidate model not promoted, underperforms current champion by {abs(delta):.4f} degC (margin={margin:.4f} degC)"

    metadata = ModelMetadata(
        model_id=f"model_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        train_mae=training_metrics["model_mae"],
        holdout_baseline_mae=baseline_mae,
        holdout_model_mae=candidate_mae,
        is_promoted=is_promoted,
        promotion_reason=reason
    )
    
    save_candidate(candidate_model, metadata, artifacts_dir)
    
    if is_promoted:
        promote_to_champion(metadata.model_id, artifacts_dir)
    
    return (metadata, is_promoted)

if __name__ == "__main__":
    from twre.features.pipeline import fetch_training_data
    from twre.models.train import split_train_holdout, train_candidate
    
    # dummy
    raw_df = fetch_training_data()
    X_train, y_train, X_holdout, y_holdout = split_train_holdout(raw_df)
    
    model = train_candidate(X_train, y_train)
    
    metadata, is_promoted = evaluate_and_promote(
        model,
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        margin=0.00
    )

    print(f"model metadata: {metadata}")
    print(f"model promoted: {is_promoted}")