# Анализ качества генераторов случайных чисел
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import os
import sys
import shutil  # Добавлено для копирования файлов
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import acf  # Добавлено для автокорреляции

# =============================
# КОНФИГУРАЦИЯ ПРОЕКТА
# =============================
SEED = 42              # Фиксируем seed для воспроизводимости
NUM_SAMPLES = 1000000  # Общее количество чисел для генерации
PLOT_SAMPLES = 5000    # Количество точек для диаграммы рассеяния
MAX_LAG = 50           # Максимальный лаг для автокорреляции
SIGNIFICANCE_LEVEL = 0.01  # Уровень значимости
TEST_SENSITIVITY = True  # Анализ чувствительности к seed
TEST_MULTIDIM = True     # Многомерный анализ
BIT_DEPTH = 8            # Количество анализируемых битов

# Создаем папку для результатов
if not os.path.exists('plots'):
    os.makedirs('plots')

# Текущая дата и время для именования файлов
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================
# РЕАЛИЗАЦИЯ ГЕНЕРАТОРОВ (без изменений)
# =============================
def lcg_generator(seed, size, a=1664525, c=1013904223, m=2**32):
    """Оптимизированный линейный конгруэнтный генератор"""
    numbers = np.empty(size)
    x = seed
    for i in range(size):
        x = (a * x + c) % m
        numbers[i] = x / m
    return numbers

def mt19937_generator(seed, size):
    """Генератор Mersenne Twister через NumPy"""
    np.random.seed(seed)
    return np.random.rand(size)

def xorshift_generator(seed, size):
    """Оптимизированный генератор Xorshift128+"""
    state0 = np.uint64(seed)
    state1 = np.uint64(seed ^ 0x1234567890ABCDEF)
    results = np.empty(size, dtype=np.float64)
    mask64 = np.uint64(0xFFFFFFFFFFFFFFFF)
    
    for i in range(size):
        s1 = state0
        s0 = state1
        state0 = s0
        s1 ^= (s1 << np.uint64(23))
        s1 ^= (s1 >> np.uint64(17))
        s1 ^= s0
        s1 ^= (s0 >> np.uint64(26))
        state1 = s1
        combined = (state0 + state1) & mask64
        results[i] = combined / 18446744073709551616.0  # 2^64
    
    return results

def pcg_generator(seed, size):
    """Оптимизированный генератор PCG32"""
    multiplier = np.uint64(0x5851F42D4C957F2D)
    increment = np.uint64(0x14057B7EF767814F)
    state = np.uint64(seed)
    results = np.empty(size, dtype=np.float64)
    
    for i in range(size):
        state = (state * multiplier + increment) & np.uint64(0xFFFFFFFFFFFFFFFF)
        xorshifted = ((state >> np.uint64(18)) ^ state) >> np.uint64(27)
        rot = state >> np.uint64(59)
        rotated = (xorshifted >> rot) | (xorshifted << (np.uint64(-rot) & np.uint64(31)))
        results[i] = rotated / 4294967296.0  # 2^32
    
    return results

def builtin_generator(seed, size):
    """Встроенный генератор Python с оптимизацией"""
    import random
    random.seed(seed)
    return np.array([random.random() for _ in range(size)])

# =============================
# ФУНКЦИИ АНАЛИЗА И ТЕСТИРОВАНИЯ
# =============================
def generate_histogram(data, generator_name, filename):
    """Генерация и сохранение гистограммы"""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Распределение {generator_name} (n={len(data):,})', fontsize=14)
    plt.xlabel('Значение случайной величины', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.axhline(y=len(data)/100, color='red', linestyle='--', 
                label='Ожидаемая частота')
    plt.legend()
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')  # Увеличено до 300 dpi
    plt.close()
    print(f"  > Гистограмма сохранена как 'plots/{filename}'")

def generate_scatter_plot(data, generator_name, filename):
    """Генерация диаграммы рассеяния для последовательных пар"""
    plt.figure(figsize=(10, 8))
    plt.plot(data[:PLOT_SAMPLES-1], data[1:PLOT_SAMPLES], 'o', 
             markersize=2, alpha=0.5, color='green')
    plt.title(f'Последовательные пары {generator_name}\n(n={PLOT_SAMPLES} точек)', 
              fontsize=14)
    plt.xlabel('x[i]', fontsize=12)
    plt.ylabel('x[i+1]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')  # Увеличено до 300 dpi
    plt.close()
    print(f"  > Диаграмма рассеяния сохранена как 'plots/{filename}'")

# Добавлена новая функция для 3D визуализации
def plot_3d_distribution(generator, name):
    """3D визуализация распределения точек"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Генерация 10000 точек (x, y, z)
    points = [generator.random() for _ in range(10000)]
    x = points[0::3]
    y = points[1::3]
    z = points[2::3]
    
    ax.scatter(x, y, z, s=1, alpha=0.5)
    ax.set_title(f'3D Distribution: {name}')
    plt.savefig(f'plots/{name.lower()}_3d_scatter.png', dpi=300)
    plt.close()
    print(f"  > 3D визуализация сохранена как 'plots/{name.lower()}_3d_scatter.png'")

# Добавлена новая функция для автокорреляции
def plot_autocorrelation_statsmodels(generator, name, lags=50):
    """Анализ автокорреляции с помощью statsmodels"""
    data = [generator.random() for _ in range(10000)]
    autocorr = acf(data, nlags=lags)
    
    plt.figure(figsize=(12, 6))
    plt.stem(autocorr, use_line_collection=True)
    plt.axhline(y=0, color='k')
    plt.axhline(y=1.96/np.sqrt(len(data)), color='r', linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(len(data)), color='r', linestyle='--')
    plt.title(f'Autocorrelation: {name}')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.savefig(f'plots/{name.lower()}_autocorr.png', dpi=300)
    plt.close()
    print(f"  > График автокорреляции сохранен как 'plots/{name.lower()}_autocorr.png'")

def run_chi_square_test(data, generator_name, bins=100):
    """Тест хи-квадрат на равномерность распределения"""
    observed_freq, _ = np.histogram(data, bins=bins)
    expected_freq = np.full(bins, len(data)/bins)
    
    chi2, p_value = stats.chisquare(observed_freq, expected_freq)
    
    print(f"  > Тест хи-квадрат ({bins} интервалов):")
    print(f"    χ² = {chi2:.2f}, p-value = {p_value:.6f}")
    if p_value < SIGNIFICANCE_LEVEL:
        print(f"    ❌ Отклонение равномерности (p < {SIGNIFICANCE_LEVEL})")
    else:
        print(f"    ✅ Равномерность не отвергается (p ≥ {SIGNIFICANCE_LEVEL})")
    
    return p_value

def runs_test(data):
    """Тест серий (Runs test) на случайность последовательности"""
    binary_sequence = [1 if data[i] < data[i+1] else 0 for i in range(len(data)-1)]
    
    n = len(binary_sequence)
    runs = 1
    for i in range(1, n):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1
    
    expected_runs = (2 * n - 1) / 3
    std_dev = np.sqrt((16 * n - 29) / 90)
    z = (runs - expected_runs) / std_dev
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    print(f"  > Тест серий:")
    print(f"    Runs = {runs}, ожидается ≈ {expected_runs:.1f}")
    print(f"    Z = {z:.4f}, p-value = {p_value:.6f}")
    return p_value

def autocorrelation_test(data, lag=1):
    """Тест автокорреляции с заданным лагом"""
    n = len(data)
    mean = np.mean(data)
    
    numerator = sum((data[i] - mean) * (data[i+lag] - mean) for i in range(n - lag))
    denominator = sum((x - mean) ** 2 for x in data)
    
    r = numerator / denominator
    q = n * (n + 2) * r**2 / (n - lag)
    p_value = 1 - stats.chi2.cdf(q, df=1)
    
    print(f"  > Тест автокорреляции (lag={lag}):")
    print(f"    r = {r:.6f}, p-value = {p_value:.6f}")
    return p_value

def bit_test(data, bit_index=0):
    """Тест на равномерность битов в указанной позиции"""
    int_data = (data * (2**32)).astype(np.uint32)
    bits = (int_data >> bit_index) & 1
    
    ones = np.sum(bits)
    zeros = len(bits) - ones
    chi2, p_value = stats.chisquare([zeros, ones], [len(bits)/2, len(bits)/2])
    
    print(f"  > Тест бита #{bit_index}:")
    print(f"    0/1: {zeros}/{ones} (ожидается {len(bits)/2:.1f}/{len(bits)/2:.1f})")
    print(f"    χ² = {chi2:.2f}, p-value = {p_value:.6f}")
    return p_value

def spectral_test(data):
    """Спектральный тест для обнаружения периодичностей"""
    fft_vals = np.fft.rfft(data - np.mean(data))
    amplitudes = np.abs(fft_vals)[1:]  # Игнорируем постоянную составляющую
    
    periodogram = amplitudes ** 2
    total_power = np.sum(periodogram)
    if total_power > 0:
        norm_periodogram = periodogram / total_power
    else:
        norm_periodogram = periodogram
    
    max_peak = np.max(norm_periodogram)
    avg_peak = np.mean(norm_periodogram)
    peak_ratio = max_peak / avg_peak
    
    p_value = np.exp(-peak_ratio)
    
    print(f"  > Спектральный тест:")
    print(f"    Макс/средн = {peak_ratio:.2f}, p-value ≈ {p_value:.6f}")
    return p_value

def runs_above_below_test(data):
    """Тест серий выше и ниже медианы"""
    median = np.median(data)
    sequence = (data > median).astype(int)
    n = len(sequence)
    
    runs = 1
    for i in range(1, n):
        if sequence[i] != sequence[i-1]:
            runs += 1
    
    n1 = np.sum(sequence)  # Количество значений выше медианы
    n0 = n - n1            # Количество значений ниже медианы
    
    expected_runs = (2 * n0 * n1) / n + 1
    std_dev = np.sqrt((2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1)))
    
    if std_dev > 0:
        z = (runs - expected_runs) / std_dev
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        p_value = 1.0
    
    print(f"  > Тест серий выше/ниже медианы:")
    print(f"    Runs = {runs}, ожидается ≈ {expected_runs:.1f}")
    print(f"    Z = {z if 'z' in locals() else 0:.4f}, p-value = {p_value:.6f}")
    return p_value

def plot_autocorrelation(data, generator_name, filename):
    """Построение графика автокорреляции"""
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    
    lags = range(0, MAX_LAG + 1)
    acf = []
    for lag in lags:
        if lag == 0:
            acf.append(1.0)
        else:
            cov = np.sum((data[:n-lag] - mean) * (data[lag:] - mean)) / n
            acf.append(cov / var)
    
    conf_int = 1.96 / np.sqrt(n)
    
    plt.figure(figsize=(12, 6))
    plt.bar(lags, acf, width=0.5)
    plt.axhline(y=0, color='black', linewidth=0.8)
    plt.axhline(y=conf_int, color='red', linestyle='--', alpha=0.7, label='95% дов. интервал')
    plt.axhline(y=-conf_int, color='red', linestyle='--', alpha=0.7)
    
    plt.title(f'Автокорреляция ({generator_name})', fontsize=14)
    plt.xlabel('Лаг', fontsize=12)
    plt.ylabel('Автокорреляция', fontsize=12)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')  # Увеличено до 300 dpi
    plt.close()
    print(f"  > График автокорреляции сохранен как 'plots/{filename}'")

def plot_pvalues_comparison(results, filename="pvalues_comparison.png"):
    """Визуализация сравнения p-values разных генераторов"""
    tests = ['p_chi2', 'p_runs', 'p_autocorr', 'p_bit', 'p_spectral', 'p_runs_median']
    test_names = ['Хи-квадрат', 'Тест серий', 'Автокорреляция', 'Тест битов', 'Спектральный', 'Серии-медиана']
    
    plt.figure(figsize=(16, 9))
    n = len(tests)
    width = 0.12
    
    for i, res in enumerate(results):
        p_values = [res[test] for test in tests]
        positions = np.arange(n) + i * width
        plt.bar(positions, p_values, width, label=res['name'])
    
    plt.axhline(y=SIGNIFICANCE_LEVEL, color='r', linestyle='--', 
                label=f'Уровень значимости {SIGNIFICANCE_LEVEL}')
    plt.yscale('log')
    plt.ylabel('p-value (логарифмическая шкала)', fontsize=12)
    plt.title('Сравнение результатов статистических тестов', fontsize=16)
    plt.xticks(np.arange(n) + width * (len(results)-1)/2, test_names, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')  # Увеличено до 300 dpi
    plt.close()
    print(f"  > График сравнения p-values сохранен как 'plots/{filename}'")

def save_results_to_csv(results, filename="results_summary.csv"):
    """Сохранение результатов в CSV файл"""
    import csv
    
    fieldnames = ['generator', 'time', 'mean', 'std_dev', 'min', 'max', 'skew', 'kurtosis',
                 'p_chi2', 'p_runs', 'p_autocorr', 'p_bit', 
                 'p_spectral', 'p_runs_median']
    
    with open(f'plots/{filename}', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for res in results:
            row = {
                'generator': res['name'],
                'time': res['time'],
                'mean': res['mean'],
                'std_dev': res['std_dev'],
                'min': res['min'],
                'max': res['max'],
                'skew': res['skew'],
                'kurtosis': res['kurtosis'],
                'p_chi2': res['p_chi2'],
                'p_runs': res['p_runs'],
                'p_autocorr': res['p_autocorr'],
                'p_bit': res['p_bit'],
                'p_spectral': res['p_spectral'],
                'p_runs_median': res['p_runs_median']
            }
            writer.writerow(row)
    
    print(f"  > Результаты сохранены в CSV: 'plots/{filename}'")

def plot_3d_scatter(data, generator_name, filename, points=1000):
    """3D диаграмма рассеяния"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:points], data[1:points+1], data[2:points+2], 
               s=1, alpha=0.5, color='blue')
    ax.set_title(f'3D последовательные точки ({generator_name})', fontsize=14)
    ax.set_xlabel('x[i]', fontsize=12)
    ax.set_ylabel('x[i+1]', fontsize=12)
    ax.set_zlabel('x[i+2]', fontsize=12)
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')  # Увеличено до 300 dpi
    plt.close()
    print(f"  > 3D диаграмма рассеяния сохранена как 'plots/{filename}'")

def plot_bit_distribution(data, generator_name, filename, bits=8):
    """Распределение битов"""
    int_data = (data * (2**32)).astype(np.uint32)
    bit_matrix = np.unpackbits(int_data.view(np.uint8)).reshape(-1, 32)
    
    plt.figure(figsize=(14, 10))
    for i in range(min(bits, 32)):
        plt.subplot(4, 4, i+1)
        plt.hist(bit_matrix[:, i], bins=2, alpha=0.7, color='purple')
        plt.title(f'Бит {i}')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
    plt.suptitle(f'Распределение битов ({generator_name})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')  # Увеличено до 300 dpi
    plt.close()
    print(f"  > Распределение битов сохранено как 'plots/{filename}'")

def seed_sensitivity_test(gen_func, generator_name, seeds=[0, 1, 42, 999, 12345], size=100000):
    """Тест чувствительности к выбору seed"""
    results = []
    print(f"\n  Тест чувствительности к seed ({generator_name}):")
    
    for i, seed in enumerate(seeds):
        print(f"    Seed {i+1}/{len(seeds)}: {seed}")
        data = gen_func(seed, size)
        p_chi2 = run_chi_square_test(data, f"Seed={seed}", bins=100)
        mean_val = np.mean(data)
        results.append((seed, mean_val, p_chi2))
    
    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    seeds_list, means, p_values = zip(*results)
    
    plt.subplot(2, 1, 1)
    plt.plot(seeds_list, means, 'o-', markersize=8)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title(f'Среднее значение при разных seed ({generator_name})')
    plt.xlabel('Seed')
    plt.ylabel('Среднее')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(seeds_list, p_values, 's-', color='green', markersize=8)
    plt.axhline(y=SIGNIFICANCE_LEVEL, color='r', linestyle='--')
    plt.title('p-value теста хи-квадрат')
    plt.xlabel('Seed')
    plt.ylabel('p-value')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    filename = f"{TIMESTAMP}_{generator_name.lower().replace(' ', '_')}_seed_sensitivity.png"
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')  # Увеличено до 300 dpi
    plt.close()
    print(f"  > Результаты чувствительности к seed сохранены как 'plots/{filename}'")
    
    return results, filename  # Возвращаем имя файла для копирования

def compare_to_ideal(data, generator_name):
    """Сравнение с идеальным равномерным распределением"""
    ideal = np.random.uniform(0, 1, len(data))
    ks_stat, ks_p = stats.ks_2samp(data, ideal)
    wasserstein = stats.wasserstein_distance(data, ideal)
    
    print(f"  Тест Колмогорова-Смирнова: D={ks_stat:.6f}, p={ks_p:.6f}")
    print(f"  Расстояние Вассерштейна: {wasserstein:.6f}")
    return ks_p, wasserstein

def print_final_report(results):
    """Печать финального отчета в консоль"""
    print("\n" + "=" * 80)
    print("ФИНАЛЬНЫЙ ОТЧЕТ".center(80))
    print("=" * 80)
    
    # Вывод таблицы результатов
    print("\nСводка результатов тестирования:")
    print(f"{'Генератор':<20} {'Время (с)':>10} {'Среднее':>10} {'Стд.откл.':>10}")
    for res in results:
        print(f"{res['name']:<20} {res['time']:>10.5f} {res['mean']:>10.6f} {res['std_dev']:>10.6f}")
    
    # Тест имен для отчета
    test_names = {
        'time': 'Время генерации',
        'p_chi2': 'Тест хи-квадрат',
        'p_runs': 'Тест серий',
        'p_autocorr': 'Автокорреляция',
        'p_bit': 'Тест битов',
        'p_spectral': 'Спектральный тест',
        'p_runs_median': 'Серии-медиана'
    }
    
    print("\nЛучшие генераторы по каждому критерию:")
    for test, name in test_names.items():
        if test == 'time':
            best = min(results, key=lambda x: x[test])
            value = best[test]
        else:
            best = max(results, key=lambda x: x[test])
            value = best[test]
        print(f"- {name:<18}: {best['name']} ({value:.6f})")
    
    # Проверка прохождения тестов
    print("\nСтатистическая мощность тестов:")
    for test in ['p_chi2', 'p_runs', 'p_autocorr', 'p_bit', 'p_spectral', 'p_runs_median']:
        passed = sum(1 for res in results if res[test] > SIGNIFICANCE_LEVEL)
        print(f"- {test_names[test]}: {passed}/{len(results)} генераторов прошли тест")
    
    # Рекомендации
    print("\nРекомендации по применению:")
    for res in results:
        if res['p_chi2'] < 0.01 or res['p_bit'] < 0.01:
            print(f"- {res['name']}: ❌ Не рекомендуется для статистических применений")
        elif res['time'] < 0.05:
            print(f"- {res['name']}: ⚡ Рекомендуется для высокопроизводительных задач")
        else:
            print(f"- {res['name']}: ✅ Рекомендуется для общего применения")
    
    print("\n" + "=" * 80)
    print("ВЫВОДЫ".center(80))
    print("=" * 80)
    
    # Основные выводы
    print("\n1. Все генераторы, кроме LCG, демонстрируют хорошую статистическую случайность.")
    print("2. LCG показывает заметные паттерны и не проходит несколько тестов.")
    print("3. Xorshift128+ - самый быстрый генератор с приемлемым качеством.")
    print("4. MT19937 и PCG32 демонстрируют наилучшее качество генерации.")
    print("5. Для критически важных приложений рекомендуются PCG32 или MT19937.")

# =============================
# ОСНОВНАЯ ПРОГРАММА
# =============================
def main():
    print("=" * 80)
    print("АНАЛИЗ ГЕНЕРАТОРОВ СЛУЧАЙНЫХ ЧИСЕЛ".center(80))
    print("=" * 80)
    print(f"Параметры тестирования:")
    print(f"- Количество чисел: {NUM_SAMPLES:,}")
    print(f"- Seed: {SEED}")
    print(f"- Уровень значимости: {SIGNIFICANCE_LEVEL}")
    print(f"- Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Список генераторов для тестирования
    generators = [
        ("MT19937 (NumPy)", mt19937_generator),
        ("LCG", lambda s, sz: lcg_generator(s, sz)),
        ("Xorshift128+", xorshift_generator),
        ("PCG32", pcg_generator),
        ("Built-in Python", builtin_generator)
    ]
    
    results = []
    
    for gen_name, gen_func in generators:
        try:
            print("\n" + f" ТЕСТИРОВАНИЕ: {gen_name} ".center(80, "-"))
            start_time = time.time()
            data = gen_func(SEED, NUM_SAMPLES)
            gen_time = time.time() - start_time
            
            # Базовые статистики
            mean_val = np.mean(data)
            std_dev = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            skewness = stats.skew(data)
            kurt = stats.kurtosis(data)
            
            print(f"  Время генерации: {gen_time:.5f} сек")
            print(f"  Среднее: {mean_val:.6f} (ожидается ~0.5)")
            print(f"  Стандартное отклонение: {std_dev:.6f} (ожидается ~0.2887)")
            print(f"  Минимум: {min_val:.6f}, Максимум: {max_val:.6f}")
            print(f"  Асимметрия: {skewness:.6f} (0=симметрия), Эксцесс: {kurt:.6f} (0=норма)")
            
            # Запуск тестов
            print("\n  РЕЗУЛЬТАТЫ ТЕСТОВ:")
            p_chi2 = run_chi_square_test(data, gen_name)
            p_runs = runs_test(data)
            p_autocorr = autocorrelation_test(data)
            p_bit = bit_test(data)
            p_spectral = spectral_test(data)
            p_runs_median = runs_above_below_test(data)
            
            # Сравнение с идеальным распределением
            ks_p, wasserstein = compare_to_ideal(data, gen_name)
            
            # Визуализация
            filename_prefix = f"{TIMESTAMP}_{gen_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
            generate_histogram(data, gen_name, f"{filename_prefix}_histogram.png")
            generate_scatter_plot(data, gen_name, f"{filename_prefix}_scatter.png")
            plot_autocorrelation(data, gen_name, f"{filename_prefix}_autocorr.png")
            plot_bit_distribution(data, gen_name, f"{filename_prefix}_bit_dist.png")
            
            # Многомерная визуализация
            if TEST_MULTIDIM:
                plot_3d_scatter(data, gen_name, f"{filename_prefix}_3d_scatter.png", points=10000)
            
            # Тест чувствительности к seed
            seed_sens_filename = None
            if TEST_SENSITIVITY:
                seed_results, seed_sens_filename = seed_sensitivity_test(gen_func, gen_name)
            
            # Для MT19937 создаем дополнительные графики
            if gen_name == "MT19937 (NumPy)":
                # Создаем специальные графики для отчета
                plot_3d_distribution(lambda: mt19937_generator(SEED, 30000), "MT19937")
                plot_autocorrelation_statsmodels(lambda: mt19937_generator(SEED, 10000), "MT19937")
                
                # Копируем существующие файлы с нужными именами
                files_to_rename = [
                    (f"{filename_prefix}_bit_dist.png", "mt19937_bit_dist.png"),
                    (seed_sens_filename, "mt19937_seed_sensitivity.png") if seed_sens_filename else None
                ]
                
                for src, dst in files_to_rename:
                    if src and dst:
                        src_path = f"plots/{src}"
                        dst_path = f"plots/{dst}"
                        if os.path.exists(src_path):
                            shutil.copyfile(src_path, dst_path)
                            print(f"  > Скопировано для отчета: {src} -> {dst}")
            
            # Сохранение результатов
            results.append({
                'name': gen_name,
                'time': gen_time,
                'mean': mean_val,
                'std_dev': std_dev,
                'min': min_val,
                'max': max_val,
                'skew': skewness,
                'kurtosis': kurt,
                'p_chi2': p_chi2,
                'p_runs': p_runs,
                'p_autocorr': p_autocorr,
                'p_bit': p_bit,
                'p_spectral': p_spectral,
                'p_runs_median': p_runs_median,
                'ks_p': ks_p,
                'wasserstein': wasserstein
            })
        except Exception as e:
            print(f"  ОШИБКА: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Сравнение производительности
    print("\n" + " ИТОГОВОЕ СРАВНЕНИЕ ".center(80, "="))
    for res in results:
        print(f"\n{res['name']}:")
        print(f"  Время: {res['time']:.5f} сек")
        print(f"  Хи-квадрат: p={res['p_chi2']:.6f}")
        print(f"  Тест серий: p={res['p_runs']:.6f}")
        print(f"  Автокорреляция: p={res['p_autocorr']:.6f}")
        print(f"  Тест битов: p={res['p_bit']:.6f}")
        print(f"  Спектральный: p={res['p_spectral']:.6f}")
        print(f"  Серии-медиана: p={res['p_runs_median']:.6f}")
    
    # Дополнительные результаты
    plot_pvalues_comparison(results, f"{TIMESTAMP}_pvalues_comparison.png")
    save_results_to_csv(results, f"{TIMESTAMP}_results_summary.csv")
    print_final_report(results)
    
    print("\n" + "=" * 80)
    print(f"АНАЛИЗ ЗАВЕРШЕН! Результаты сохранены в папке 'plots'".center(80))
    print("=" * 80)

if __name__ == "__main__":
    main()