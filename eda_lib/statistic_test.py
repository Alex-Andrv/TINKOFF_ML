import numpy as np
from scipy import stats

def shapiro_test(data):
    # Тест Шапиро-Уилка
    shapiro_test_statistic, shapiro_p_value = stats.shapiro(data)
    print("Тест Шапиро-Уилка:")
    print(f"Статистика теста: {shapiro_test_statistic}")
    print(f"p-значение: {shapiro_p_value}")
    if shapiro_p_value > 0.05:
        print("Данные похожи на нормальное распределение.")
    else:
        print("Данные не похожи на нормальное распределение.")

def anderson_test(data):
    # Тест Андерсона-Дарлинга
    anderson_test_statistic, anderson_critical_values, _ = stats.anderson(data)
    print("\nТест Андерсона-Дарлинга:")
    print(f"Статистика теста: {anderson_test_statistic}")
    print(f"Критические значения: {anderson_critical_values}")
    if anderson_test_statistic < anderson_critical_values[2]:
        print("Данные похожи на нормальное распределение.")
    else:
        print("Данные не похожи на нормальное распределение.")

def ks_test(data, dist='norm'):
    # Тест Колмогорова-Смирнова
    ks_test_statistic, ks_p_value = stats.kstest(data, dist)
    print("\nТест Колмогорова-Смирнова:")
    print(f"Статистика теста: {ks_test_statistic}")
    print(f"p-значение: {ks_p_value}")
    if ks_p_value > 0.05:
        print(f"Данные похожи на {dist} распределение.")
    else:
        print(f"Данные не похожи на {dist} распределение.")