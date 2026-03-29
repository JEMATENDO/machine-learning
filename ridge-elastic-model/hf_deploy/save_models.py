# ============================================================
# Run this script (or paste as a cell) AFTER training your
# models in the notebook, so that ridge_model, lasso_model,
# scaler, and FEATURE_COLS are already in memory.
# ============================================================
import joblib, os, json

os.makedirs("hf_deploy", exist_ok=True)

joblib.dump(ridge_model, "hf_deploy/ridge_model.pkl")
joblib.dump(lasso_model, "hf_deploy/lasso_model.pkl")
joblib.dump(scaler,      "hf_deploy/scaler.pkl")

with open("hf_deploy/feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)

print("Saved:")
print("  hf_deploy/ridge_model.pkl")
print("  hf_deploy/lasso_model.pkl")
print("  hf_deploy/scaler.pkl")
print("  hf_deploy/feature_cols.json")
