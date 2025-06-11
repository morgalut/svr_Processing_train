import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

def load_test_data(data_dir):
    npz_path = os.path.join(data_dir, 'X_test.npz')
    print(f"📂 Loading NPZ from: {npz_path}")

    # Load sparse matrix properly
    X_test = sparse.load_npz(npz_path)   # 🛠️ Correct for sparse
    print(f"✅ Loaded sparse matrix with shape: {X_test.shape}")

    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    print(f"✅ Loaded y_test with shape: {y_test.shape}")

    return X_test, y_test



def safe_to_dense(X, max_elements=1e8):
    if sparse.issparse(X):
        num_elements = X.shape[0] * X.shape[1]
        if num_elements > max_elements:
            raise MemoryError(f"Matrix too large to densify: {num_elements} elements.")
        return X.toarray()
    return X


def evaluate_model(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def plot_predictions(y_true, y_pred, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True CTR')
    plt.ylabel('Predicted CTR')
    plt.title('CTR Prediction: True vs Predicted')
    plt.grid(True)

    # Set finer ticks
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    tick_step = 0.02
    ticks = np.arange(0, max_val + tick_step, tick_step)
    plt.xticks(ticks)
    plt.yticks(ticks)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📈 Plot saved to {save_path}")
    # No plt.show()
    plt.close()


def fill_nan_dense(X):
    return np.nan_to_num(X)

def assert_no_nan(X, name="input"):
    if np.isnan(X).any():
        raise ValueError(f"🚨 {name} still contains NaNs after preprocessing!")

def main():
    data_dir = os.path.join("processed")
    model_dir = os.path.join("svr")
    output_dir = os.path.join("test_results")
    os.makedirs(output_dir, exist_ok=True)

    print("📦 Loading test data...")
    X_test, y_test = load_test_data(data_dir)

    print("📦 Loading trained SVR model...")
    feature_scaler = joblib.load(os.path.join(model_dir, "feature_scaler.joblib"))
    svr_model = joblib.load(os.path.join(model_dir, "linear_svr_model.joblib"))

    # Preprocess test data
    X_test_dense = safe_to_dense(X_test)
    X_test_dense = fill_nan_dense(X_test_dense)      # 💊 Fix NaNs
    assert_no_nan(X_test_dense, "X_test_dense")      # ✅ Sanity check
    X_test_scaled = feature_scaler.transform(X_test_dense)

    print("🧠 Predicting CTR on test data...")
    y_pred = svr_model.predict(X_test_scaled)

    print("📊 Evaluating model...")
    metrics = evaluate_model(y_test, y_pred)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    plot_path = os.path.join(output_dir, "svr_ctr_true_vs_predicted.png")
    plot_predictions(y_test, y_pred, save_path=plot_path)

if __name__ == "__main__":
    main()
