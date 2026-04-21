import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# --- Data loading and preprocessing ---
def load_and_preprocess_data(file_path, target_column):
    file = pd.read_csv(file_path).dropna()
    X = file.drop(columns=target_column)
    y = file[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# --- Helpers ---

def enforce_sensitive_difference(x, x_prime, X_ref, sensitive_columns):
    for col in sensitive_columns:
        possible = [v for v in X_ref[col].unique() if v != x[col]]
        if possible:
            x_prime[col] = random.choice(possible)
        else:
            x_prime[col] = x[col]
    return x, x_prime


def hash_individual(x):
    return tuple(x.values.tolist())

# --- Initialization ---

def initialise_population(size, X_test, sensitive_columns):
    population = []
    for _ in range(size):
        x = X_test.sample(n=1).iloc[0].copy()
        x_prime = x.copy()
        x, x_prime = enforce_sensitive_difference(x, x_prime, X_test, sensitive_columns)
        population.append((x, x_prime))
    return population


# --- Fitness ---

def evaluate_fitness_batch(model, population, X_ref, non_sensitive_columns):
    xs = np.array([ind[0].values for ind in population])
    xps = np.array([ind[1].values for ind in population])

    preds_x = model.predict(xs, verbose=0).flatten()
    preds_xp = model.predict(xps, verbose=0).flatten()

    fitnesses = []
    for i, ind in enumerate(population):
        fitnesses.append(abs(preds_x[i] - preds_xp[i]))
    return fitnesses


# --- Selection ---

def select_parents(population, fitnesses, num_parents, tournament_size=3):
    selected = []
    paired = list(zip(population, fitnesses))
    for _ in range(num_parents):
        tournament = random.sample(paired, tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected


# --- Crossover ---

def crossover(parent1, parent2, sensitive_columns, non_sensitive_columns, X_test):
    if random.random() > 0.8:
        return parent1, parent2

    (x_a, x_a_prime), (x_b, x_b_prime) = parent1, parent2

    def make_child(base, x1, x2):
        child = base.copy()

        # mix non-sensitive features
        for col in non_sensitive_columns:
            child[col] = random.choice([x1[col], x2[col]])

        return pd.Series(child)

    #generate children
    child1_x = make_child(x_a, x_a, x_b)
    child1_x_prime = make_child(x_a_prime, child1_x, child1_x)

    child2_x = make_child(x_b, x_b, x_a)
    child2_x_prime = make_child(x_b_prime, child2_x, child2_x)

    child1_x, child1_x_prime = enforce_sensitive_difference(child1_x, child1_x_prime, X_test, sensitive_columns)
    child2_x, child2_x_prime = enforce_sensitive_difference(child2_x, child2_x_prime, X_test, sensitive_columns)

    return (child1_x, child1_x_prime), (child2_x, child2_x_prime)


# --- Mutation ---

def mutation(child, X_test, non_sensitive_columns):
    x, x_prime = child
    mutation_rate = 1/len(non_sensitive_columns)  # Dynamically adjust mutation rate based on number of features
    mutation_rate *= 1  #Manually adjust mutation rate to control exploration

    for col in non_sensitive_columns:
        if random.random() > mutation_rate:
            continue

        min_v, max_v = X_test[col].min(), X_test[col].max()
        range_v = max_v - min_v

        #same perturbation for x and x'
        mutation_strength = 0.1 #Adjust mutation strength to control exploration
        perturb = np.random.uniform(-mutation_strength * range_v, mutation_strength * range_v)

        new_val_x = np.clip(x[col] + perturb, min_v, max_v)
        new_val_xp = np.clip(x_prime[col] + perturb, min_v, max_v)

        if np.issubdtype(type(x[col]), np.integer):
            new_val_x = int(round(new_val_x))
            new_val_xp = int(round(new_val_xp))

        x[col] = new_val_x
        x_prime[col] = new_val_xp

    return x, x_prime


# --- Replacement ---

def replace_population(population, fitnesses, children):
    combined = [(ind, fit) for ind, fit in zip(population, fitnesses)] + children
    combined.sort(key=lambda x: x[1], reverse=True)
    new_pop = combined[:len(population)]
    return [ind for ind, fit in new_pop], [fit for ind, fit in new_pop]

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

def genetic_algorithm_evaluation(model, population_size, X_test, sensitive_columns, non_sensitive_columns, budget, idi_threshold=0.05, checkpoints=None): 
    if checkpoints is None:
        checkpoints = [100, 200, 400, 600, 800, 1000]
 
    population = initialise_population(population_size, X_test, sensitive_columns)
    fitnesses = evaluate_fitness_batch(model, population, X_test, non_sensitive_columns)
 
    seen_inputs = set()
    IDI_set = set()
 
    idi_severities  = []
    ckpt_log        = []
    first_idi_eval  = None
    checkpoint_idx  = 0
 
    def register(individual):
        x, x_prime = individual
        pair_hash = (hash_individual(x), hash_individual(x_prime))
        if pair_hash in seen_inputs:
            return 0, pair_hash
        seen_inputs.add(pair_hash)
        return 1, pair_hash
    
    #Initial population severity evaluation
    x_init  = np.array([ind[0].values for ind in population])
    xp_init = np.array([ind[1].values for ind in population])
    p_x_init  = model.predict(x_init,  verbose=0).flatten()
    p_xp_init = model.predict(xp_init, verbose=0).flatten()
 
    #Log IDIs and severities for initial population
    for i, (ind, fit) in enumerate(zip(population, fitnesses)):
        new, pair_hash = register(ind)
        if not new:
            continue
 
        sev = abs(p_x_init[i] - p_xp_init[i])
        if sev > idi_threshold:
            IDI_set.add(pair_hash)
            idi_severities.append(sev)
            if first_idi_eval is None:
                first_idi_eval = len(seen_inputs)
 
        if checkpoint_idx < len(checkpoints) and len(seen_inputs) >= checkpoints[checkpoint_idx]:
            ckpt_log.append(len(IDI_set))
            checkpoint_idx += 1
 
    # --- Genetic loop ---
    while len(seen_inputs) < budget:
        children = []
        parents = select_parents(population, fitnesses, population_size // 2)
        pending_children = []
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                break
            child1, child2 = crossover(parents[i], parents[i+1], sensitive_columns, non_sensitive_columns, X_test)
            child1 = mutation(child1, X_test, non_sensitive_columns)
            child2 = mutation(child2, X_test, non_sensitive_columns)
 
            for child in [child1, child2]:
                if len(seen_inputs) >= budget:
                    break
                new, pair_hash = register(child)
                if not new:
                    continue
                pending_children.append((child, pair_hash))
 
        if pending_children:
            child_only      = [c for c, _ in pending_children]
            child_hashes    = [h for _, h in pending_children]
            child_fitnesses = evaluate_fitness_batch(model, child_only, X_test, non_sensitive_columns)
 
            xs_ch  = np.array([c[0].values for c in child_only])
            xps_ch = np.array([c[1].values for c in child_only])
            p_x_ch  = model.predict(xs_ch,  verbose=0).flatten()
            p_xp_ch = model.predict(xps_ch, verbose=0).flatten()
 
            for j, (child, pair_hash) in enumerate(zip(child_only, child_hashes)):
                fit = child_fitnesses[j]
 
                sev = abs(p_x_ch[j] - p_xp_ch[j])
                if sev > idi_threshold:
                    IDI_set.add(pair_hash)
                    idi_severities.append(sev)
                    if first_idi_eval is None:
                        first_idi_eval = len(seen_inputs)
 
                if checkpoint_idx < len(checkpoints) and len(seen_inputs) >= checkpoints[checkpoint_idx]:
                    ckpt_log.append(len(IDI_set))
                    checkpoint_idx += 1
 
                children.append((child, fit))
 
        if not children:
            break
 
        population, fitnesses = replace_population(population, fitnesses, children)
 
    #fill any checkpoints not yet reached
    while len(ckpt_log) < len(checkpoints):  # ADDED
        ckpt_log.append(len(IDI_set))
 
    best_idx = np.argmax(fitnesses)
    best_solution = population[best_idx]
 
    idi_ratio = len(IDI_set) / max(len(seen_inputs), 1)
 
    half_idi_eval = estimate_half_idi(ckpt_log, checkpoints, len(IDI_set))
 
    return best_solution, idi_ratio, idi_severities, ckpt_log, first_idi_eval, half_idi_eval

DATASETS = {
    "adult":             ("Class-label", ["race", "gender", "age"]),
    "communities_crime": ("class",       ["Black", "FemalePctDiv"]),
    "compas":            ("Recidivism",  ["Sex", "Race"]),
    "credit":            ("class",       ["SEX", "EDUCATION", "MARRIAGE"]),
    "dutch":             ("occupation",  ["sex", "age"]),
    "german":            ("CREDITRATING",["PersonStatusSex", "AgeInYears"]),
    "kdd":               ("income",      ["race", "sex"]),
    "law_school":        ("pass_bar",    ["male", "race"]),
}

def main():
    dataset_name = "compas"  #change this to run on different datasets
    target_column, sensitive_columns = DATASETS[dataset_name]
 
    file_path  = f"dataset/processed_{dataset_name}.csv"
    model_path = f"DNN/model_processed_{dataset_name}.h5"
    model = load_model(model_path)
 
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_column)
 
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]
    population_size = 50
 
    best_solution, idi_ratio, idi_severities, ckpt_log, first_idi_eval, half_idi_eval = genetic_algorithm_evaluation(model, population_size, X_test, sensitive_columns, non_sensitive_columns, budget=1000)
 
    print(f"Best solution found: {best_solution}")
    print(f"Prediction for x: {model.predict(best_solution[0].values.reshape(1, -1))[0][0]}")
    print(f"Prediction for x': {model.predict(best_solution[1].values.reshape(1, -1))[0][0]}")
    fitness = evaluate_fitness_batch(model, [best_solution], X_test, non_sensitive_columns)[0]
    print(f"Fitness of best solution: {fitness}")

    #print metrics
    print(f"IDI ratio        : {idi_ratio:.4f}")
    print(f"Mean severity    : {np.mean(idi_severities):.4f}" if idi_severities else "Mean severity    : N/A")
    print(f"Max severity     : {np.max(idi_severities):.4f}"  if idi_severities else "Max severity     : N/A")
    print(f"First IDI at eval: {first_idi_eval}")
    print(f"Half IDI at eval : {half_idi_eval}")
    print(f"Checkpoint log   : {ckpt_log}")

if __name__ == "__main__":
    main()