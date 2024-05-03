"""Элементы калькулятора параметров A/B-теста."""
import numpy as np

from scipy.stats import norm


class ABTestSampleSize:
    """Класс расчета размера выборки для A/B-теста."""

    def __init__(self, mde:float=0.05, alpha:float=0.05, beta:float=0.2, two_sided:bool=True, k:int=1) -> None:
        """Конструктор класса.
        Аргументы (определение; диапазон значений; дефолтное значение):
           - mde - абсолютный (относительный для mean) минимальный детектируемый эффект; [0, 1.0]; 0.05
           - alpha - вероятность ошибки I рода; [0, 1.0]; 0.05
           - beta - вероятность ошибки II рода; [0, 1.0]; 0.2
           - two_sided - флаг двустороннего теста; [False, True]; True
           - k - количество тестовых групп; [1, inf); 1
        """
        self.input_data = {'MDE': mde, 'Alpha': alpha, 'Beta': beta, 'TwoSided': two_sided}
        self.k = k
        self.mde = mde
        self.z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
        self.z_beta = norm.ppf(1 - beta)
        self.n_coeff = 1 + np.sqrt(k)
        self.m_coeff = k + np.sqrt(k)

    def get_sample_size_for_mean(self, mean:float, sigma:float):
        """Функция вычисляет размер выборки в случае, когда метрика - среднее.
        Аргументы (определение; диапазон значений):
           - mean - текущее среднее значение; (-inf, inf)
           - sigma - среднеквадратичное отклонение (генеральной совокупности); [0, inf)
        Возвращаемое значение (массив [n, m]):
           - n - минимальный статистически значимый размер контрольной выборки; [0, inf)
           - m - минимальный статистически значимый размер одной из k тестовых выборок; [0, inf)
        """
        delta = mean * self.mde
        total = np.ceil(
            (self.n_coeff + self.m_coeff) * (sigma * (self.z_alpha + self.z_beta) / delta) ** 2
        )
        
        self.data = np.array([total / self.n_coeff, total / self.m_coeff]).astype(int), 'mean'
        return self.data

    def get_sample_size_for_conversion(self, conversion:float, is_em_calc:bool=True):
        """Функция вычисляет размер выборки в случае, когда метрика - конверсия.
        Аргументы (определение; диапазон значений):
           - conversion - текущий уровень конверсии; [0, 1.0]
           - is_em_calc - флаг использования методики evanmiller калькулятора; [False, True]; True
        Возвращаемое значение (массив [n, m]):
           - n - минимальный статистически значимый размер контрольной выборки; [0, inf)
           - m - минимальный статистически значимый размер одной из k тестовых выборок; [0, inf)
        """
        delta = self.mde
        sample_rate = self.m_coeff / self.n_coeff
        conv_control = conversion
        conv_test = conversion + delta
        conv_average = conv_control if is_em_calc else (conv_control + sample_rate * conv_test) / (1 + sample_rate)
        n = np.ceil(((
            np.sqrt(conv_average * (1 - conv_average) * (1 + 1 / sample_rate)) * self.z_alpha +
            np.sqrt(conv_control * (1 - conv_control) + conv_test * (1 - conv_test) / sample_rate) * self.z_beta
        ) / delta) ** 2)

        self.data = np.array([n, n * sample_rate]).astype(int), 'conversion'
        return self.data
    
    def __str__(self) -> str:
        args = '\n'.join(key + ': ' + str(val) for key, val in self.input_data.items())
        
        return (
            f'____________________________________________________________________________________\n'
            f'Используемая метрика: {self.data[1]}\n'
            f'{args}\n'
            f'____________________________________________________________________________________\n'
            f'Колличество контрольных/тестовых групп: 1/{self.k}\n'
            f'Количество объектов в контрольной/тестовой группе: {self.data[0][0]}/{self.data[0][1]}\n'
            f'____________________________________________________________________________________\n'
        )    
