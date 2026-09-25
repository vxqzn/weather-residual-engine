import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

from twre.models.gate import evaluate_and_promote
from twre.models.registry import load_champion, load_ledger

@pytest.fixture
def synthetic_data():
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    })
    y = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    
    return X, y, X, y

def test_cold_start_promotion(tmp_path, synthetic_data):
    X_train, y_train, X_holdout, y_holdout = synthetic_data

    # cold start scenario, no champion model exists
    perfect_model = DummyRegressor(strategy="constant", constant=2.0)
    perfect_model.fit(X_train, y_train)
    
    metadata, is_promoted = evaluate_and_promote(
        perfect_model,
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        artifacts_dir=tmp_path
    )

    assert is_promoted is True    
    assert (tmp_path / "ledger.json").exists(), "ledger file does not exist"
    assert load_ledger(tmp_path)["active_model"] == metadata.model_id, "active model id in ledger does not match candidate id"
    
def test_degraded_candidate_is_rejected(tmp_path, synthetic_data):    
    X_train, y_train, X_holdout, y_holdout = synthetic_data
    
    # degraded candidate scenario, champion model exists, candidate is worse
    champignon = DummyRegressor(strategy="constant", constant=1.0)
    champignon.fit(X_train, y_train)
    champignon_metadata, _ = evaluate_and_promote(
        champignon,
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        artifacts_dir=tmp_path
    )
    
    degraded = DummyRegressor(strategy="constant", constant=555.0)
    degraded.fit(X_train, y_train)
    degraded_metadata, is_promoted = evaluate_and_promote(
        degraded,
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        artifacts_dir=tmp_path
    )
    
    assert is_promoted is False
    assert champignon_metadata.is_promoted is True
    assert load_ledger(tmp_path)["active_model"] == champignon_metadata.model_id, "active model id in ledger should match champion id (no promo)"
    assert "not promoted" in degraded_metadata.promotion_reason.lower(), "promotion reason should indicate candidate was not promoted"
    
def test_superior_candidate_is_promoted(tmp_path, synthetic_data):
    X_train, y_train, X_holdout, y_holdout = synthetic_data
    
    # superior candidate scenario, champion model exists, candidate is better
    champignon = DummyRegressor(strategy="constant", constant=1.0)
    champignon.fit(X_train, y_train)
    champignon_metadata, _ = evaluate_and_promote(
        champignon,
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        artifacts_dir=tmp_path
    )
    
    good_model = DummyRegressor(strategy="constant", constant=1.999)
    good_model.fit(X_train, y_train)
    good_metadata, is_promoted = evaluate_and_promote(
        good_model,
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        artifacts_dir=tmp_path
    )
    
    assert is_promoted is True
    assert load_ledger(tmp_path)["active_model"] == good_metadata.model_id, "active model id in ledger should match candidate id (promo)"