import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Подход 1: градиентный бустинг "в лоб"

# 1. Считайте таблицу с признаками из файла features.csv
# Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке)
train = pd.read_csv("data/final/features.csv", index_col="match_id")
train.drop([
    "duration",
    "tower_status_radiant",
    "tower_status_dire",
    "barracks_status_radiant",
    "barracks_status_dire",
            ], axis=1, inplace=True)


# 2. Проверьте выборку на наличие пропусков с помощью функции count(),
# которая для каждого столбца показывает число заполненных значений.
# Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски,
# и попробуйте для любых двух из них дать обоснование, почему их значения могут быть пропущены.
count_na = len(train) - train.count()
passes = count_na[count_na > 0].sort_values(ascending=False) / len(train)


# 3. Замените пропуски на нули с помощью функции fillna()
train.fillna(0, inplace=True)


# 4. Какой столбец содержит целевую переменную?
# Запишите его название.
X_train = train.drop("radiant_win", axis=1)
y_train = train["radiant_win"]


# Обучене градиентного бустинга над деревьями на имеющейся матрице "объекты-признаки"
cv = KFold(n_splits=5, shuffle=True, random_state=42)


def score_gb(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}

    for n_estimators in [10, 20, 30, 50, 100, 250]:
        print(f"n_estimators={n_estimators}")
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

        start_time = datetime.datetime.now()
        score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.3f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")

        scores[n_estimators] = score
        print()

    return pd.Series(scores)


# scores = score_gb(X_train, y_train)


# Подход 2: логистическая регрессия

# 1. Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией)
# с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)


def score_lr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}

    for i in range(-5, 6):
        C = 10.0 ** i

        print(f"C={C}")
        model = LogisticRegression(solver='lbfgs', C=C, random_state=42)

        start_time = datetime.datetime.now()
        score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.3f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")

        scores[i] = score
        print()

    return pd.Series(scores)


scores = score_lr(X_train, y_train)


def print_best_lr_score(scores: pd.Series):
    best_iteration = scores.sort_values(ascending=False).head(1)
    best_C = 10.0 ** best_iteration.index[0]
    best_score = best_iteration.values[0]

    print(f"Наилучшее значение показателя AUC-ROC достигается при C = {best_C:.2f} и равно {best_score:.2f}.")


print_best_lr_score(scores)


# 2. Среди признаков в выборке есть категориальные,
# которые мы использовали как числовые, что вряд ли является хорошей идеей

hero_columns = [f"r{i}_hero" for i in range (1, 6)] + [f"d{i}_hero" for i in range (1, 6)]
cat_columns = ["lobby_type"] + hero_columns
X_train.drop(cat_columns, axis=1, inplace=True)

scores = score_lr(X_train, y_train)
print_best_lr_score(scores)


# 3. На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают,
# какие именно герои играли за каждую команду

unique_heroes = np.unique(train[hero_columns].values.ravel())
N = max(unique_heroes)
print(f"Число уникальных героев в train: {len(unique_heroes)}. Максимальный ID героя: {N}.")


# 4. Воспользуемся подходом "мешок слов" для кодирования информации о героях
def get_pick(data: pd.DataFrame) -> pd.DataFrame:
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in range(1, 6):
            X_pick[i, data.loc[match_id, f"r{p}_hero"] - 1] = 1
            X_pick[i, data.loc[match_id, f"d{p}_hero"] - 1] = -1

    return pd.DataFrame(X_pick, index=data.index, columns=[f"hero_{i}" for i in range(N)])


X_pick = get_pick(train)
X_train = pd.concat([X_train, X_pick], axis=1)

# 5. Проведите кросс-валидацию для логистической регрессии на новой выборке
# с подбором лучшего параметра регуляризации

scores = score_lr(X_train, y_train)
print_best_lr_score(scores)


# 6. Постройте предсказания вероятностей победы команды Radiant для тестовой выборки
# с помощью лучшей из изученных моделей (лучшей с точки зрения AUC-ROC на кросс-валидации)

model = LogisticRegression(solver='lbfgs', C=0.1, random_state=42)
model.fit(X_train, y_train)

test = pd.read_csv("data/final/features_test.csv", index_col="match_id")
test.fillna(0, inplace=True)

X_test = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
X_test.drop(cat_columns, axis=1, inplace=True)
X_test = pd.concat([X_test, get_pick(test)], axis=1)

preds = pd.Series(model.predict_proba(X_test)[:, 1])
print(preds.describe())
