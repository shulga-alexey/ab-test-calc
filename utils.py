"""Элементы калькулятора параметров A/B-теста."""
import datetime
import config
import clickhouse_connect
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
        if is_em_calc and conversion > 0.5:
            conversion = 1 - conversion

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
        """Функция возвращает результат расчета размера выборки A/B-теста в строковом представлении."""
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


class ABTestDuration:
    """Класс расчета продолжительности A/B-теста."""

    AMOUNT_COLOUMN = 'amount'
    AMOUNT_PRED_COLOUMN = 'amount_pred'
    DATE_COLOUMN = 'date'

    def __init__(self, sample_size: int, query: str, seasonal_cycle: int=0) -> None:
        """Конструктор класса.
        Аргументы (определение; диапазон значений; дефолтное значение):
           - sample_size - количество событий, которое необходимо набрать в выборку; [0, inf)
           - query - запрос в базу данных для создания временного ряда, ожидаемые поля - date, amount;
                     здесь временной ряд представляет собой посуточное распределение amount (количество событий)
           - seasonal_cycle - период сезона в данных (количество  записей), где 0 - отсутствие сезонности; [0, inf); 0
        """
        self.sample_size = sample_size
        self.seasonal_cycle = seasonal_cycle

        time_series = self.__query_result(query)
        train_size = int(len(time_series) * 0.8)
        self.train, self.test = time_series[0: train_size], time_series[train_size: len(time_series)]

        self.optimal_params = self.__search_optimal_params()
        if self.seasonal_cycle:
            model = SARIMA(
                self.train,
                order=self.optimal_params['optimal_order_param'],
                seasonal_order=self.optimal_params['optimal_seasonal_param'],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            model = ARIMA(
                self.train,
                order=self.optimal_params['optimal_order_param'],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        self.model_fit = model.fit()

        self.minimum_days = self.__minimum_days()

    def __query_result(self, query) -> pd.DataFrame:
        """Возвращает результат запроса к базе данных.
        Возвращаемое значение:
           - pandas.DataFrame текущего запроса к базе данных
        """
        client = clickhouse_connect.get_client(
            host=config.HOST,
            port=config.PORT,
            username=config.USERNAME,
            password=config.PASSWORD
        )
        result = client.query_df(query)
        result.set_index(self.DATE_COLOUMN, inplace=True)

        return result

    def __search_optimal_params(self) -> dict:
        """Функция осуществляет поиск оптимальных параметров используемой модели.
        Возвращаемое значение (словарь {optimal_order_param, optimal_seasonal_param, smallest_aic}):
           - optimal_order_param - оптимальный p, d, q параметр (массив)
           - optimal_seasonal_param - оптимальный сезонный параметр (массив)
           - smallest_aic - оптимальный aic, который можно получить для данной модели
        """
        smallest_aic = float("inf")
        order_vals = diff_vals = ma_vals = range(0, 3)

        pdq_combinations = list(itertools.product(order_vals, diff_vals, ma_vals))
        seasonal_combinations = [(combo[0], combo[1], combo[2], self.seasonal_cycle) for combo in pdq_combinations]
        optimal_order_param = None
        optimal_seasonal_param = None

        if self.seasonal_cycle:
            for order_param in pdq_combinations:
                for seasonal_param in seasonal_combinations:
                    try:
                        sarima_model = SARIMAX(
                            self.train,
                            order=order_param,
                            seasonal_order=seasonal_param,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        model_results = sarima_model.fit()
                        if model_results.aic < smallest_aic:
                            smallest_aic = model_results.aic
                            optimal_order_param = order_param
                            optimal_seasonal_param = seasonal_param
                    except:
                        continue
        else:
            for order_param in pdq_combinations:
                try:
                    arima_model = ARIMA(
                        self.train,
                        order=order_param,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    model_results = arima_model.fit()
                    if model_results.aic < smallest_aic:
                        smallest_aic = model_results.aic
                        optimal_order_param = order_param
                except:
                    continue

        return {
            'optimal_order_param': optimal_order_param,
            'optimal_seasonal_param': optimal_seasonal_param,
            'smallest_aiс': smallest_aic
        }

    def __minimum_days(self) -> int:
        """Функция возвращает количество дней, которое требуется для набора выборки."""
        forecast_future = self.model_fit.forecast(steps=500)[len(self.train):]

        sum_amount, num_days = 0, 0
        for index, amount in sorted(forecast_future.items(), key=(lambda x: x[0])):
            sum_amount += amount
            num_days += 1
            if sum_amount >= self.sample_size:
                break

        return num_days

    def adfuller_test(self) -> None:
        """Функция возвращает результат теста Дики-Фуллера на стационарность временного ряда."""
        adfuller_result = adfuller(self.train[self.AMOUNT_COLOUMN])

        print('ADF Statistic: %f' % adfuller_result[0])
        print('p-value: %f' % adfuller_result[1])
        print('Critical Values:')
        for key, value in adfuller_result[4].items():
            print('\t%s: %.3f' % (key, value))

    def show(self) -> None:
        """Функция выводит в консоль подробную информацию по фиттированию, временной ряд и прогнозируемый эффект."""
        print(self.model_fit.summary())

        self.model_fit.plot_diagnostics(figsize=(12, 8))
        plt.show()

        start = self.train.index.max() + datetime.timedelta(days=1)
        periods = len(self.test) + self.minimum_days

        forecast = self.model_fit.get_forecast(steps=periods)
        forecast_df = pd.DataFrame({
            self.DATE_COLOUMN: pd.date_range(start=start, periods=periods, freq='D'),
            self.AMOUNT_PRED_COLOUMN: forecast.predicted_mean
        })
        forecast_df.set_index('date', inplace=True)

        # Calculate the mean squared error
        mse = mean_squared_error(self.test, forecast_df[:self.test.index.max()])
        rmse = mse ** 0.5

        # Create a plot to compare the forecast with the actual test data
        plt.figure(figsize=(12, 8))
        plt.plot(self.train, label='Training Data')
        plt.plot(self.test, label='Actual Data', color='orange')
        plt.plot(forecast_df, label='Forecasted Data', color='green')
        plt.fill_between(forecast_df.index,
                         forecast.conf_int().iloc[:, 0],
                         forecast.conf_int().iloc[:, 1],
                         color='k', alpha=0.05)
        plt.title('ARIMA Model Evaluation')
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.legend()
        plt.show()

        print('RMSE:', rmse)

    def __str__(self) -> str:
        """Функция возвращает результат расчета продолжительности A/B-теста."""

        return (
            f'____________________________________________________________________________________\n'
            f'Всего необходимо набрать событий: {self.sample_size}\n'
            f'Ожидаемое время набора выборки, если начать сегодня: {self.minimum_days} дней\n'
            f'____________________________________________________________________________________\n'
        )
