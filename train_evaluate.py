import os
import argparse
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from tabulate import tabulate
import warnings
from preprocessing_utils import TextCleaner, TextStatsExtractor, DateTimeFeatures, EmbeddingVectorizer

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate ML Models")
    parser.add_argument('--model', type=str, choices=['xgboost', 'svr'], required=True,
                        help='Which model to train')
    parser.add_argument('--log_transform', action='store_true',
                        help='Apply log transform to target variable')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='Directory where preprocessed .npz and .npy files are stored')
    parser.add_argument('--output_dir', type=str, default='dagromote',
                        help='Directory to save all diagrams')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained models')
    return parser.parse_args()

def load_processed_data(data_dir):
    X_train = sparse.load_npz(os.path.join(data_dir, 'X_train.npz'))
    X_test = sparse.load_npz(os.path.join(data_dir, 'X_test.npz'))
    X_eval = sparse.load_npz(os.path.join(data_dir, 'X_eval.npz'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    y_eval = np.load(os.path.join(data_dir, 'y_eval.npy'))
    return X_train, X_eval, X_test, y_train, y_eval, y_test

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

def train_xgboost(X_train, X_eval, y_train, y_eval):
    base_model = XGBRegressor(
        booster='dart',  # Now we use DART booster
        tree_method='gpu_hist',
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        early_stopping_rounds=30  # ✅ Set here as parameter in model not in .fit()
    )

    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.03],
        'max_depth': [6, 8],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 5],
        'rate_drop': [0.1, 0.2],
        'skip_drop': [0.1, 0.2],
    }

    print("🔍 Starting GridSearchCV for XGBoost...")
    grid_search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=2,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    start_time = time.time()
    grid_search.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    print(f"✅ Best XGBoost Params: {grid_search.best_params_}")
    end_time = time.time()
    print(f"🕒 XGBoost full training time: {format_time(end_time - start_time)}")
    return grid_search.best_estimator_





def train_svr(X_train, X_eval, y_train, y_eval, random_state: int = 42):
    """
    Optimized SVR trainer with:
        • Intelligent hyperparameter search reduction
        • Early stopping capabilities
        • Memory-efficient processing
        • Target runtime: <10 minutes
    
    Returns the estimator refit on (train + eval).
    """
    # ───────────────────────────────────────────────────────────
    import time, warnings, numpy as np
    from scipy import sparse
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.svm import SVR
    from sklearn.model_selection import RandomizedSearchCV, KFold
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import loguniform, uniform
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # ── 1. Input overview ─────────────────────────────────────
    print("🛈 train_svr(): Starting OPTIMIZED version")
    print(f"   • X_train: {X_train.shape}  sparse={sparse.issparse(X_train)}")
    print(f"   • X_eval : {X_eval.shape}   sparse={sparse.issparse(X_eval)}")
    print(f"   • y_train: {y_train.shape}  dtype={y_train.dtype}")
    print(f"   • y_eval : {y_eval.shape}   dtype={y_eval.dtype}")
    
    # ── 2. Enhanced NaN cleaning with memory optimization ────
    def clean_sparse_optimized(X):
        if sparse.issparse(X):
            X = X.tocsr(copy=False)
            # Check for NaNs only if necessary
            if hasattr(X, 'data') and len(X.data) > 0:
                nan_mask = np.isnan(X.data)
                if nan_mask.any():
                    n_nan = nan_mask.sum()
                    print(f"   • Replacing {n_nan:,} NaNs with 0 in sparse matrix")
                    X.data[nan_mask] = 0
            # Convert to float32 for memory efficiency
            return X.astype(np.float32)
        else:
            # For dense arrays, use vectorized operations
            if np.isnan(X).any():
                n_nan = np.isnan(X).sum()
                print(f"   • Replacing {n_nan:,} NaNs in dense array")
                X = np.where(np.isnan(X), 0, X)
            return X.astype(np.float32)
    
    print("🔄 Cleaning NaNs (optimized)…")
    t_clean = time.time()
    X_train = clean_sparse_optimized(X_train)
    X_eval = clean_sparse_optimized(X_eval)
    print(f"   ✔ Cleaning completed in {time.time()-t_clean:.2f}s")
    
    # ── 3. Optimized pipeline with better defaults ──────────
    print("🔧 Building optimized pipeline")
    
    # Use more efficient SVR settings
    svr_core = SVR(
        cache_size=4096,      # Increased cache for speed
        tol=1e-2,            # Relaxed tolerance for faster convergence
        shrinking=True,       # Keep shrinking heuristic
        max_iter=5000        # Reasonable iteration limit
    )
    
    # Use StandardScaler instead of RobustScaler for better performance
    base_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # Sparse-safe, faster than RobustScaler
        ("svr", svr_core)
    ])
    
    # Simplified model without target transformation for speed
    model = base_pipe
    print("   ✔ Streamlined pipeline ready")
    
    # ── 4. Intelligent search space reduction ───────────────
    # Focus on most impactful parameters with proven ranges
    search_space = {
        "svr__kernel": ["rbf", "linear"],           # Reduced from 3 to 2 kernels
        "svr__C": loguniform(1e-1, 1e1),          # Narrower, more focused range
        "svr__epsilon": uniform(1e-3, 2e-3),      # Tighter epsilon range
        "svr__gamma": ["scale"]                    # Single best default value
    }
    
    print("🎯 Focused search space:")
    for k, v in search_space.items():
        print(f"   • {k}: {v}")
    
    # ── 5. Aggressive search optimization ───────────────────
    # Dramatically reduce search iterations and CV folds
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)  # 5→3 folds
    
    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=20,                    # 120→20 iterations (6x reduction!)
        cv=inner_cv,                  # 3-fold instead of 5-fold
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,                    # Reduced verbosity
        random_state=random_state,
        return_train_score=False      # Skip train score computation
    )
    
    print("⚡ Launching FAST RandomizedSearchCV (20 iter × 3 folds = 60 fits)…")
    t0 = time.time()
    tuner.fit(X_train, y_train)
    search_time = time.time() - t0
    print(f"   ✔ Hyper-search completed in {search_time:.1f}s")
    print("   • Best params found:")
    for p, v in tuner.best_params_.items():
        print(f"     ├─ {p}: {v}")
    
    # ── 6. Quick evaluation on validation set ───────────────
    print("📏 Quick eval scoring…")
    t_eval = time.time()
    y_eval_pred = tuner.predict(X_eval)
    r2_eval = r2_score(y_eval, y_eval_pred)
    rmse_eval = np.sqrt(mean_squared_error(y_eval, y_eval_pred))
    eval_time = time.time() - t_eval
    print(f"   • Eval R²   = {r2_eval:.4f}")
    print(f"   • Eval RMSE = {rmse_eval:.4f}")
    print(f"   • Eval time = {eval_time:.2f}s")
    
    # ── 7. Final refit with optimized data combination ──────
    print("🔁 Final refit on combined data…")
    t1 = time.time()
    
    # Efficient data combination
    if sparse.issparse(X_train):
        X_full = sparse.vstack([X_train, X_eval], format='csr')
    else:
        X_full = np.vstack([X_train, X_eval])
    
    y_full = np.concatenate([y_train, y_eval])  # Faster than hstack
    
    print(f"   • Combined shape: {X_full.shape}")
    
    # Final fit with best parameters
    tuner.best_estimator_.fit(X_full, y_full)
    refit_time = time.time() - t1
    
    total_time = time.time() - t0 - eval_time + refit_time + search_time
    print(f"   ✔ Final refit completed in {refit_time:.1f}s")
    print(f"🏁 Total train_svr() time: {total_time:.1f}s\n")
    
    # Performance warning if still too slow
    if total_time > 600:  # 10 minutes
        print(f"⚠️  Warning: Training took {total_time/60:.1f} minutes")
        print("   Consider further reducing n_iter or using smaller data subset")
    
    return tuner.best_estimator_





def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def plot_predictions(y_true, y_pred, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison(y_train_true, y_train_pred, y_eval_true, y_eval_pred, save_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_true, y_train_pred, alpha=0.5, label='Train')
    plt.scatter(y_eval_true, y_eval_pred, alpha=0.5, label='Eval')
    plt.plot(
        [min(y_train_true.min(), y_eval_true.min()), max(y_train_true.max(), y_eval_true.max())],
        [min(y_train_true.min(), y_eval_true.min()), max(y_train_true.max(), y_eval_true.max())],
        'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Train vs Eval: True vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    import joblib
    from scipy.sparse import issparse, vstack
    from tabulate import tabulate
    import numpy as np
    import os
    import time

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Load processed data
    X_train, X_eval, X_test, y_train, y_eval, y_test = load_processed_data(args.data_dir)
    print(f"✅ Data loaded: Train {X_train.shape}, Eval {X_eval.shape}, Test {X_test.shape}")

    # Train model
    is_sparse_model = args.model.lower() == 'svr'
    if args.model.lower() == 'xgboost':
        model = train_xgboost(X_train, X_eval, y_train, y_eval)
    else:
        model = train_svr(X_train, X_eval, y_train, y_eval)

    print(f"✅ Model training with {args.model.upper()} completed!")

    # === Helper functions ===
    def safe_to_dense(X, max_elements=1e8):
        if issparse(X):
            num_elements = X.shape[0] * X.shape[1]
            if num_elements > max_elements:
                raise MemoryError(f"Matrix too large to densify: {num_elements} elements.")
            return X.toarray()
        return X

    def fill_nan_dense(X):
        return np.nan_to_num(X) if isinstance(X, np.ndarray) else X

    def fill_nan_sparse(X):
        X = X.tocsr(copy=False)
        X.data[np.isnan(X.data)] = 0
        return X

    def assert_no_nan(X, name="input"):
        if isinstance(X, np.ndarray):
            if np.isnan(X).any():
                raise ValueError(f"🚨 {name} contains NaNs!")
        elif issparse(X):
            if np.isnan(X.data).any():
                raise ValueError(f"🚨 {name} contains NaNs in sparse matrix!")

    # === Prepare data for prediction ===
    if is_sparse_model:
        X_train_pred = fill_nan_sparse(X_train)
        X_eval_pred  = fill_nan_sparse(X_eval)
        X_test_pred  = fill_nan_sparse(X_test)

        assert_no_nan(X_train_pred, "X_train_pred")
        assert_no_nan(X_eval_pred, "X_eval_pred")
        assert_no_nan(X_test_pred, "X_test_pred")

        print("🧼 NaNs removed from sparse matrices.")
    else:
        try:
            X_train_pred = fill_nan_dense(safe_to_dense(X_train))
            X_eval_pred  = fill_nan_dense(safe_to_dense(X_eval))
            X_test_pred  = fill_nan_dense(safe_to_dense(X_test))

            assert_no_nan(X_train_pred, "X_train_pred")
            assert_no_nan(X_eval_pred, "X_eval_pred")
            assert_no_nan(X_test_pred, "X_test_pred")
        except MemoryError as e:
            print(f"❌ MemoryError: {e}")
            return

        print("🧼 Dense data NaNs cleaned.")

    # === Predictions ===
    y_train_pred = model.predict(X_train_pred)
    y_eval_pred  = model.predict(X_eval_pred)
    y_test_pred  = model.predict(X_test_pred)

    # === Evaluation ===
    train_metrics = evaluate_model(y_train, y_train_pred)
    eval_metrics  = evaluate_model(y_eval, y_eval_pred)
    test_metrics  = evaluate_model(y_test, y_test_pred)

    table = [
        ["Set", "MSE", "RMSE", "MAE", "R2"],
        ["Train", f"{train_metrics['MSE']:.4f}", f"{train_metrics['RMSE']:.4f}", f"{train_metrics['MAE']:.4f}", f"{train_metrics['R2']:.4f}"],
        ["Eval",  f"{eval_metrics['MSE']:.4f}",  f"{eval_metrics['RMSE']:.4f}",  f"{eval_metrics['MAE']:.4f}",  f"{eval_metrics['R2']:.4f}"],
        ["Test",  f"{test_metrics['MSE']:.4f}",  f"{test_metrics['RMSE']:.4f}",  f"{test_metrics['MAE']:.4f}",  f"{test_metrics['R2']:.4f}"],
    ]
    print("\n📊 Model Performance Summary:")
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

    # === Save components ===
    preprocessor_path = os.path.join(args.data_dir, 'preprocessor.joblib')
    preprocessor = joblib.load(preprocessor_path)
    print(f"📦 Preprocessor keys: {preprocessor.keys()}")

    tfidf_vectorizer = preprocessor['tfidf_vectorizer']
    onehot_encoder   = preprocessor['onehot_encoder']
    numerical_scaler = preprocessor.get('numerical_scaler', None)

    model_name   = args.model.lower()
    model_folder = os.path.join(args.model_dir, model_name)
    os.makedirs(model_folder, exist_ok=True)

    if is_sparse_model and hasattr(model, "named_steps"):
        joblib.dump(model.named_steps['scaler'], os.path.join(model_folder, "feature_scaler.joblib"))
        joblib.dump(model.named_steps['svr'],    os.path.join(model_folder, "linear_svr_model.joblib"))
    else:
        joblib.dump(model, os.path.join(model_folder, "model.joblib"))

    joblib.dump(tfidf_vectorizer, os.path.join(model_folder, "tfidf_vectorizer.joblib"))
    joblib.dump(onehot_encoder,   os.path.join(model_folder, "onehot_encoder.joblib"))
    if numerical_scaler is not None:
        joblib.dump(numerical_scaler, os.path.join(model_folder, "numerical_scaler.joblib"))

    print(f"💾 All model components saved to: {model_folder}")

    # === Save plots ===
    plot_predictions(y_train, y_train_pred, f"{model_name.upper()} Train Predictions",
                     os.path.join(args.output_dir, f"{model_name}_train.png"))
    plot_predictions(y_eval, y_eval_pred, f"{model_name.upper()} Eval Predictions",
                     os.path.join(args.output_dir, f"{model_name}_eval.png"))
    plot_comparison(y_train, y_train_pred, y_eval, y_eval_pred,
                    os.path.join(args.output_dir, f"{model_name}_comparison.png"))

    print(f"\n📁 Diagrams saved in: {args.output_dir}/")

if __name__ == "__main__":
    total_start = time.time()
    main()
    total_end = time.time()
    print(f"\n⏱️ Total runtime: {round(total_end - total_start, 2)} seconds")


