# ✅ ** How to Install Requirements**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```


### ✅ ** Preprocessing Step**

First preprocess the raw CSV files into `processed_data/`:

```bash
python preprocess.py data/Sport_part1.csv --output_dir processed --train_ratio 0.75 --eval_ratio 0.1 --test_ratio 0.15

```

This will create:

```
processed_data/
├── X_train.npz
├── X_test.npz
├── X_eval.npz
├── y_train.npy
├── y_test.npy
├── y_eval.npy
└── preprocessor.joblib
```

---

### ✅ ** Train Models — Specify Data and Output Directory**

#### 🚀 **Train with XGBoost**

```bash
python train_evaluate.py --model xgboost --data_dir processed_data --output_dir dagromote
```

#### 🚀 **Train with SVR**

```bash
python train_evaluate.py --model svr --data_dir processed --output_dir dagromote
```

---

### ✅ ** Train with Log Transform**

(When you want to apply `log1p()` transform to the targets)

#### 🚀 **XGBoost + Log Transform**

```bash
python train_evaluate.py --model xgboost --data_dir processed --output_dir dagromote --log_transform
```


#### 🚀 **SVR + Log Transform**

```bash
python train_evaluate.py --model svr --data_dir processed --output_dir dagromote --log_transform 
```

---

# Run without sample weights (recommended first)
```sh
python train_evaluate.py --model xgboost
```
# Run with weights for comparison
```bash
python train_evaluate.py --model xgboost --use_weights
```
# 🧩 **Folder Structure After Training**

After running, `dagromote/` will contain:

```
dagromote/
├── xgboost_model.joblib
├── xgboost_train.png
├── xgboost_eval.png
├── xgboost_comparison.png
├── catboost_model.joblib
├── catboost_train.png
├── catboost_eval.png
├── catboost_comparison.png
├── svr_model.joblib
├── svr_train.png
├── svr_eval.png
├── svr_comparison.png
```

---

✅ **All commands now include:**

* `--data_dir processed_data`
* `--output_dir dagromote`
* Optional `--log_transform` for log target

