import importlib
import sys
import types

def test_module_imports_and_attributes():
    # Import the module; importing runs top-level training -- ensure it does not crash
    bp = importlib.import_module('Burnout_Prediction')

    # Check expected top-level attributes exist
    for attr in ('model', 'le_target', 'X_test', 'y_test', 'full_df', 'feature_cols'):
        assert hasattr(bp, attr), f"Burnout_Prediction missing attribute: {attr}"


def test_model_predicts_on_test_sample():
    bp = importlib.import_module('Burnout_Prediction')

    model = bp.model
    X_test = bp.X_test
    le = bp.le_target

    # take a small sample (first row) and ensure prediction runs
    sample = X_test.iloc[[0]]
    pred = model.predict(sample)
    assert len(pred) == 1

    # ensure that inverse_transform maps to a string label without error
    label = le.inverse_transform(pred)
    assert isinstance(label[0], str)


def test_full_df_has_required_features():
    bp = importlib.import_module('Burnout_Prediction')
    df = bp.full_df
    # Ensure common columns exist
    required = ['work_hours', 'sleep_hours', 'burnout_risk']
    for c in required:
        assert c in df.columns, f"Expected column {c} in dataframe"
