import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Data loading and preprocessing
def load_and_preprocess_data(file_path, target_column):
    file = pd.read_csv(file_path).dropna()
    X = file.drop(columns=target_column)
    y = file[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# 3. Generate test sample pairs (Random Search)
def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns):
    x = X_test.sample(n=1).iloc[0]
    x_prime = x.copy()
    for col in sensitive_columns:
        pos_values = [v for v in X_test[col].unique() if v != x[col]]
        x_prime[col] = np.random.choice(pos_values)

    for col in non_sensitive_columns:
        minValue = X_test[col].min()
        maxValue = X_test[col].max()
        perturbation = np.random.uniform(-0.1 * (maxValue - minValue), 0.1 * (maxValue - minValue))
        x[col] = int(np.clip(x[col] + perturbation, minValue, maxValue))
        x_prime[col] = int(np.clip(x_prime[col] + perturbation, minValue, maxValue))
    
    return x, x_prime

# 4. Model prediction and individual discrimination evaluation
def evaluate_discrimination(model, sample_a, sample_b, threshold=0.05, discrimination_pairs=None):
    pred_a = model.predict(sample_a.values.reshape(1, -1))[0][0]
    pred_b = model.predict(sample_b.values.reshape(1, -1))[0][0]
    
    if abs(pred_a - pred_b) > threshold:
        pair_hash = (tuple(sample_a.values), tuple(sample_b.values))
        if pair_hash not in discrimination_pairs:
            discrimination_pairs.add(pair_hash)
            return True
    return False

def estimate_half_idi(ckpt_log, checkpoints, total_idis):
    if total_idis == 0:
        return None
    half = total_idis / 2
    for i, count in enumerate(ckpt_log):
        if count >= half:
            if i == 0:
                return checkpoints[0]
            prev_count = ckpt_log[i - 1]
            prev_ckpt  = checkpoints[i - 1]
            curr_ckpt  = checkpoints[i]
            if count == prev_count:
                return curr_ckpt
            frac = (half - prev_count) / (count - prev_count)
            return int(prev_ckpt + frac * (curr_ckpt - prev_ckpt))
    return checkpoints[-1]

def calculate_idi_ratio_evaluation(model, X_test, sensitive_columns, non_sensitive_columns, num_samples, checkpoints=None):
    if checkpoints is None:
        checkpoints = [100, 200, 400, 600, 800, 1000]
 
    discrimination_pairs = set()
    idi_severities       = []
    ckpt_log             = []
    first_idi_eval       = None
    checkpoint_idx       = 0
 
    for i in range(num_samples):
        x, x_prime = generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns)
 
        pred_x  = model.predict(x.values.reshape(1, -1),       verbose=0)[0][0]
        pred_xp = model.predict(x_prime.values.reshape(1, -1), verbose=0)[0][0]
        severity  = abs(pred_x - pred_xp) #fitness
        pair_hash = (tuple(x.values), tuple(x_prime.values))
 
        if severity > 0.05 and pair_hash not in discrimination_pairs:
            discrimination_pairs.add(pair_hash)
            idi_severities.append(severity)
            if first_idi_eval is None:
                first_idi_eval = i + 1
 
        eval_count = i + 1
        if checkpoint_idx < len(checkpoints) and eval_count >= checkpoints[checkpoint_idx]:
            ckpt_log.append(len(discrimination_pairs))
            checkpoint_idx += 1
 
    #fill any checkpoints beyond num_samples
    while len(ckpt_log) < len(checkpoints):
        ckpt_log.append(len(discrimination_pairs))
 
    idi_ratio     = len(discrimination_pairs) / num_samples
    half_idi_eval = estimate_half_idi(ckpt_log, checkpoints, len(discrimination_pairs))
 
    return idi_ratio, idi_severities, ckpt_log, first_idi_eval, half_idi_eval
 
# 6. Main function
def main():
    paths = ["adult", "communities_crime", "compas", "credit", "dutch", "german", "kdd", "law_school"]
    targets = ['Class-label', 'class', 'Recidivism', 'class', 'occupation', 'CREDITRATING', 'income', 'pass_bar']
    all_sensitive_columns = [
        ['race', 'gender', 'age'], ['Black', 'FemalePctDiv'], ['Sex', 'Race'],
        ['SEX', 'EDUCATION', 'MARRIAGE'], ['sex', 'age'], ['PersonalStatusSex', 'AgeInYears'],
        ['race', 'sex'], ['male', 'race'],
    ]
 
    target_column     = targets[0]
    sensitive_columns = all_sensitive_columns[0]
    data              = paths[0]
    num_samples       = 1000
 
    file_path  = "dataset/processed_" + data + ".csv"
    model_path = "DNN/model_processed_" + data + ".h5"
 
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_column)
    model = load_model(model_path)
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]
 
    idi_ratio, idi_severities, ckpt_log, first_idi_eval, half_idi_eval = calculate_idi_ratio_evaluation(model, X_test, sensitive_columns, non_sensitive_columns, num_samples)
 
    print(f"IDI Ratio        : {idi_ratio:.4f}")
    print(f"Mean Severity    : {np.mean(idi_severities):.4f}" if idi_severities else "Mean Severity    : N/A")
    print(f"Max Severity     : {np.max(idi_severities):.4f}"  if idi_severities else "Max Severity     : N/A")
    print(f"First IDI at eval: {first_idi_eval}")
    print(f"Half IDI at eval : {half_idi_eval}")
    print(f"Checkpoint log   : {ckpt_log}")

if __name__ == "__main__":
    main()