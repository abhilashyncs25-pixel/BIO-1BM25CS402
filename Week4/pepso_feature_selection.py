"""
PEPSO-inspired Feature Selection
--------------------------------
Educational Python reimplementation inspired by:
"Deng et al., Particle Swarm Optimization Algorithm for Feature Selection
Inspired by Peak Ecosystem Dynamics", Computers, Materials & Continua, 2025.

IMPORTANT:
This is not the authors' official source code or an exact reproduction.
It implements the paper's broad ideas: Information Gain + ReliefF prescreening,
binary particle-swarm search, fitness-ranked pyramid levels, winner/loser learning,
and peak-guided mutation.

CSV format:
    First, install dependencies:
        pip install numpy pandas scikit-learn

    Put your dataset in a CSV file. The target/class column must be present.
    Example:
        python pepso_feature_selection.py --data gene_data.csv --target class

Categorical input columns are automatically one-hot encoded.
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def load_dataset(csv_path, target_column):
    """Load CSV, separate target, encode categorical columns, and impute missing values."""
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[target_column]).copy()
    y_raw = df[target_column]
    X_raw = df.drop(columns=[target_column])

    # Convert categorical input columns into numeric one-hot columns.
    X = pd.get_dummies(X_raw, dummy_na=True)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove columns that are entirely missing.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns were found.")

    # Fill missing feature values using each column's median.
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    # Encode class labels as integers.
    y = LabelEncoder().fit_transform(y_raw.astype(str))

    # Scale columns to [0, 1] for ReliefF distances.
    values = X.to_numpy(dtype=float)
    mins = np.min(values, axis=0)
    maxs = np.max(values, axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    X_scaled = (values - mins) / ranges

    return X, X_scaled, y


def relief_f_scores(X, y, n_neighbors=5, random_state=42):
    """
    Simple ReliefF-style feature scoring for numeric data.
    Higher scores indicate features that help separate classes.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape
    classes = np.unique(y)

    if len(classes) < 2:
        return np.zeros(n_features)

    k = min(n_neighbors, max(1, n_samples - 1))
    weights = np.zeros(n_features, dtype=float)

    # Limit the number of sampled rows for speed on larger datasets.
    sample_count = min(n_samples, 300)
    sample_indices = rng.choice(n_samples, size=sample_count, replace=False)

    class_counts = {c: np.sum(y == c) for c in classes}
    class_probs = {c: class_counts[c] / n_samples for c in classes}

    for idx in sample_indices:
        distances = np.sqrt(np.sum((X - X[idx]) ** 2, axis=1))
        distances[idx] = np.inf

        same_class = np.where(y == y[idx])[0]
        same_class = same_class[same_class != idx]
        if len(same_class):
            hit_ids = same_class[np.argsort(distances[same_class])[:min(k, len(same_class))]]
            hit_diff = np.mean(np.abs(X[hit_ids] - X[idx]), axis=0)
            weights -= hit_diff / sample_count

        # Penalize differences to near-miss examples, weighted by class probability.
        other_classes = [c for c in classes if c != y[idx]]
        for c in other_classes:
            miss_ids_all = np.where(y == c)[0]
            if len(miss_ids_all) == 0:
                continue
            miss_ids = miss_ids_all[np.argsort(distances[miss_ids_all])[:min(k, len(miss_ids_all))]]
            miss_diff = np.mean(np.abs(X[miss_ids] - X[idx]), axis=0)
            denominator = max(1e-12, 1.0 - class_probs[y[idx]])
            weights += (class_probs[c] / denominator) * miss_diff / sample_count

    return weights


def prescreen_features(X, y, feature_names, keep_features=100):
    """
    Combine mutual information (Information-Gain-style ranking) and ReliefF.
    For continuous features, mutual_info_classif is used as a practical
    information-theoretic ranking measure.
    """
    n_features = X.shape[1]
    keep_features = min(n_features, max(1, keep_features))

    mi = mutual_info_classif(
        X, y, discrete_features=False, random_state=RANDOM_STATE
    )
    relief = relief_f_scores(X, y, random_state=RANDOM_STATE)

    # Rank-normalize both scores, then average so scales do not dominate.
    def rank_normalize(scores):
        order = np.argsort(np.argsort(scores))
        if len(scores) <= 1:
            return np.ones_like(scores, dtype=float)
        return order / (len(scores) - 1)

    combined = 0.5 * rank_normalize(mi) + 0.5 * rank_normalize(relief)
    chosen = np.argsort(combined)[::-1][:keep_features]

    print(f"\nPrescreening: {n_features} original features -> {len(chosen)} candidates")
    return chosen, combined


class PEPSOFeatureSelector:
    """
    Binary PEPSO-inspired optimizer.

    A particle is a 0/1 vector:
        1 = select feature
        0 = do not select feature

    The fitness combines cross-validation accuracy and a small penalty for
    selecting too many features. Pyramid levels are rebuilt by sorting particles
    by fitness; lower-ranked particles learn from better-ranked particles.
    """

    def __init__(
        self,
        n_particles=12,
        n_iterations=20,
        max_features=30,
        feature_penalty=0.01,
        mutation_rate=0.05,
        random_state=42,
        cv_folds=5,
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.max_features = max_features
        self.feature_penalty = feature_penalty
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.best_mask_ = None
        self.best_score_ = -np.inf
        self.history_ = []

    def _fitness(self, mask, X, y):
        selected = np.flatnonzero(mask)
        if len(selected) == 0:
            return 0.0

        X_subset = X[:, selected]
        classifier = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
        )

        class_counts = np.bincount(y)
        smallest_class = int(class_counts.min())
        folds = min(self.cv_folds, smallest_class)
        if folds < 2:
            # Very small class counts: use a single holdout evaluation.
            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, y, test_size=0.25, random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            classifier.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, classifier.predict(X_test))
        else:
            cv = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=self.random_state
            )
            scores = cross_val_score(
                classifier, X_subset, y, cv=cv, scoring="accuracy", n_jobs=1
            )
            accuracy = float(np.mean(scores))

        # Reward accuracy and mildly prefer smaller feature subsets.
        size_ratio = len(selected) / X.shape[1]
        return accuracy - self.feature_penalty * size_ratio

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        if n_features == 0:
            raise ValueError("PEPSO received zero candidate features.")

        # Initial particles select features with moderate probability.
        positions = (rng.random((self.n_particles, n_features)) < 0.30).astype(int)
        for i in range(self.n_particles):
            if not positions[i].any():
                positions[i, rng.integers(n_features)] = 1

        velocities = rng.normal(0, 0.5, size=(self.n_particles, n_features))
        personal_best = positions.copy()
        personal_fitness = np.array([self._fitness(p, X, y) for p in positions])

        best_idx = int(np.argmax(personal_fitness))
        self.best_mask_ = personal_best[best_idx].copy()
        self.best_score_ = float(personal_fitness[best_idx])

        for iteration in range(self.n_iterations):
            # Pyramid topology: high-fitness particles are at higher levels.
            order = np.argsort(personal_fitness)[::-1]
            ranked_positions = personal_best[order]

            progress = iteration / max(1, self.n_iterations - 1)
            inertia = 0.9 - 0.5 * progress

            for rank, particle_idx in enumerate(order):
                current = positions[particle_idx].copy()

                # Select a guide from a better-ranked particle where possible.
                if rank == 0:
                    guide = self.best_mask_.copy()
                else:
                    better_rank = rng.integers(0, rank)
                    guide = ranked_positions[better_rank].copy()

                # Winner/loser cooperation: weaker particles learn from better
                # particles; stronger particles keep more of their own direction.
                if rank < max(1, self.n_particles // 3):
                    learning = 1.2
                    social = 1.2
                elif rank < max(2, 2 * self.n_particles // 3):
                    learning = 1.5
                    social = 1.5
                else:
                    learning = 1.8
                    social = 1.8

                r1 = rng.random(n_features)
                r2 = rng.random(n_features)
                velocities[particle_idx] = (
                    inertia * velocities[particle_idx]
                    + learning * r1 * (personal_best[particle_idx] - current)
                    + social * r2 * (guide - current)
                )

                # Binary PSO: convert velocity into selection probabilities.
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(
                    velocities[particle_idx], -10, 10
                )))
                new_position = (rng.random(n_features) < probabilities).astype(int)

                # Peak-guided mutation: perturb some bits of top-level particles
                # occasionally to preserve diversity and explore new subsets.
                if rank < max(1, self.n_particles // 4):
                    if rng.random() < 0.30:
                        flips = rng.random(n_features) < self.mutation_rate
                        new_position[flips] = 1 - new_position[flips]

                if not new_position.any():
                    new_position[rng.integers(n_features)] = 1

                # Limit excessively large subsets by keeping strongest velocity
                # dimensions, while ensuring at least one feature remains.
                if self.max_features and new_position.sum() > self.max_features:
                    selected_ids = np.flatnonzero(new_position)
                    strongest = selected_ids[
                        np.argsort(velocities[particle_idx, selected_ids])[
                            -self.max_features:
                        ]
                    ]
                    new_position[:] = 0
                    new_position[strongest] = 1

                positions[particle_idx] = new_position
                score = self._fitness(new_position, X, y)

                if score > personal_fitness[particle_idx]:
                    personal_best[particle_idx] = new_position.copy()
                    personal_fitness[particle_idx] = score

                if score > self.best_score_:
                    self.best_score_ = float(score)
                    self.best_mask_ = new_position.copy()

            self.history_.append(self.best_score_)
            print(
                f"Iteration {iteration + 1:02d}/{self.n_iterations} | "
                f"best fitness={self.best_score_:.4f} | "
                f"selected={int(self.best_mask_.sum())}"
            )

        return self


def main():
    parser = argparse.ArgumentParser(
        description="PEPSO-inspired feature selection for a labeled CSV dataset."
    )
    parser.add_argument("--data", required=True, help="Path to dataset CSV file")
    parser.add_argument("--target", required=True, help="Target/class column name")
    parser.add_argument("--keep", type=int, default=100,
                        help="Number of features to retain after prescreening (default: 100)")
    parser.add_argument("--particles", type=int, default=12,
                        help="Number of particles (default: 12)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Optimization iterations (default: 20)")
    parser.add_argument("--max-features", type=int, default=30,
                        help="Maximum selected candidate features (default: 30)")
    parser.add_argument("--output", default="selected_features.csv",
                        help="Output file for selected features")
    args = parser.parse_args()

    X_df, X_scaled, y = load_dataset(args.data, args.target)
    feature_names = np.asarray(X_df.columns)

    if len(np.unique(y)) < 2:
        raise ValueError("The target column must contain at least two classes.")

    # Prescreen using information-theoretic and ReliefF-style scores.
    candidate_ids, _ = prescreen_features(
        X_scaled, y, feature_names, keep_features=args.keep
    )
    X_candidates = X_scaled[:, candidate_ids]
    candidate_names = feature_names[candidate_ids]

    selector = PEPSOFeatureSelector(
        n_particles=args.particles,
        n_iterations=args.iterations,
        max_features=args.max_features,
        random_state=RANDOM_STATE,
    )
    selector.fit(X_candidates, y)

    selected_local = np.flatnonzero(selector.best_mask_)
    selected_original_names = candidate_names[selected_local]
    selected_original_ids = candidate_ids[selected_local]

    print("\n========== RESULTS ==========")
    print(f"Original feature count: {X_df.shape[1]}")
    print(f"Prescreened candidate count: {len(candidate_ids)}")
    print(f"Selected feature count: {len(selected_local)}")
    print(f"Best fitness (accuracy minus size penalty): {selector.best_score_:.4f}")
    print("\nSelected features:")
    for name in selected_original_names:
        print(f" - {name}")

    # Export the selected feature subset plus target labels.
    output_df = X_df.iloc[:, selected_original_ids].copy()
    output_df[args.target] = y
    output_df.to_csv(args.output, index=False)
    print(f"\nSelected dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
