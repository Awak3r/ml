# Лабораторная работа №1: линейная регрессия

В [ноутбуке `lab1.ipynb`](lab1/lab1.ipynb) загружаем данные, строим графики, вызываем `fit` и `predict`, сравниваем метрики и сохраняем CSV. Основные алгоритмы вынесены в `.py`-файлы.

| Что | Где написано |
|---|---|
| Очистка данных и one-hot | [`clean.py`](lab1/src/clean.py) |
| Z-score и min-max | [`scalers.py`](lab1/src/scalers.py) |
| МНК, GD, SGD и регуляризация | [`linear.py`](lab1/src/linear.py) |
| MSE, MAE, R² и MAPE | [`metrics.py`](lab1/src/metrics.py) |
| Учебные K-fold и LOO | [`cv.py`](lab1/src/cv.py) |
| Подбор порогов, две ветки и расчёт их веса | [`branch_mixture.py`](lab1/src/branch_mixture.py) |
| Кросс-валидация модели для Kaggle | [`validate_risk.py`](lab1/validate_risk.py) |

Идея двух веток и выбор исходных полей основаны на [решении liptonMartin](https://github.com/liptonMartin/mai_labs_machine_learning) (`lab_1/01_train_sklearn.ipynb`). Числовые пороги подбираются на обучающих данных.
