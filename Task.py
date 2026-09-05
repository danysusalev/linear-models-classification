import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        self.categories_ = []
        for column in X.columns:
            unique_vals = sorted(X[column].unique())
            self.categories_.append(unique_vals)

    def transform(self, X):
        total_features = sum(len(cat) for cat in self.categories_)
        result = np.zeros((X.shape[0], total_features), dtype=self.dtype)
        current_index = 0

        for i, column in enumerate(X.columns):
            categories = self.categories_[i]
            mapping = {val: idx for idx, val in enumerate(categories)}
            indices = X[column].map(mapping).dropna().astype(int)
            valid_rows = indices.index
            result[valid_rows, current_index + indices] = 1
            current_index += len(categories)

        return result

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        self.stats_ = {}
        n_objects = len(X)

        for column in X.columns:
            column_stats = {}
            for value in X[column].unique():
                mask = X[column] == value
                count = mask.sum()

                if count > 0:
                    success = Y[mask].mean()
                    counter = count / n_objects
                else:
                    success = 0.0
                    counter = 0.0

                column_stats[value] = (success, counter)

            self.stats_[column] = column_stats

    def transform(self, X, a=1e-5, b=1e-5):
        n_objects = len(X)
        n_features = len(X.columns)
        result = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for col_idx, column in enumerate(X.columns):
            for row_idx, value in enumerate(X[column]):
                if value in self.stats_[column]:
                    success, counter = self.stats_[column][value]
                    relation = (success + a) / (counter + b)
                else:
                    success, counter, relation = 0.0, 0.0, 0.0

                result[row_idx, 3 * col_idx] = success
                result[row_idx, 3 * col_idx + 1] = counter
                result[row_idx, 3 * col_idx + 2] = relation

        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_ : (i + 1) * n_], np.hstack(
            (idx[: i * n_], idx[(i + 1) * n_ :])
        )
    yield idx[(n_splits - 1) * n_ :], idx[: (n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        n_objects = len(X)
        self.fold_stats_ = []
        self.fold_assignments_ = np.zeros(n_objects, dtype=int)

        folds = list(group_k_fold(n_objects, self.n_folds, seed))

        for fold_idx, (test_idx, train_idx) in enumerate(folds):
            self.fold_assignments_[test_idx] = fold_idx

            X_train = X.iloc[train_idx]
            Y_train = Y.iloc[train_idx]

            fold_stat = {}
            for column in X.columns:
                column_stats = {}

                for value in X_train[column].unique():
                    mask = X_train[column] == value
                    count = mask.sum()

                    if count > 0:
                        success = Y_train[mask].mean()
                        counter = count / len(train_idx)
                    else:
                        success = 0.0
                        counter = 0.0

                    column_stats[value] = (success, counter)

                fold_stat[column] = column_stats

            self.fold_stats_.append(fold_stat)

    def transform(self, X, a=1e-5, b=1e-5):
        n_objects = len(X)
        n_features = len(X.columns)
        result = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for row_idx in range(n_objects):
            fold_idx = self.fold_assignments_[row_idx]
            fold_stat = self.fold_stats_[fold_idx]

            for col_idx, column in enumerate(X.columns):
                value = X[column].iloc[row_idx]

                if column in fold_stat and value in fold_stat[column]:
                    success, counter = fold_stat[column][value]
                    relation = (success + a) / (counter + b)
                else:
                    success, counter, relation = 0.0, 0.0, 0.0

                result[row_idx, 3 * col_idx] = success
                result[row_idx, 3 * col_idx + 1] = counter
                result[row_idx, 3 * col_idx + 2] = relation

        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    unique_vals = np.sort(np.unique(x))

    optimal_weights = []
    for val in unique_vals:
        mask = x == val

        if np.sum(mask) > 0:
            weight = np.mean(y[mask])
        else:
            weight = 0.0

        optimal_weights.append(weight)

    return np.array(optimal_weights)
