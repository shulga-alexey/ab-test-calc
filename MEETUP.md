# Improving AB-testing

## Методы уменьшения дисперсии исследуемой метрики

По причине нехватки данных или желания бизнеса сократить издержки при проведении A/B-тестов появляется необходимость в уменьшении выборки, на которой разыгрывается тест. Тогда, сократив время набора данных за счет уменьшения их объема можно раньше внедрить успешные изменения в проект или же раньше откатить провальные фичи. 
  
Напомним упрощенную формулу, по которой можно найти минимально необходимый объем требующихся данных:

```math 
n = {&sigma;^2 Z^2 \over m^2}
```
где $&sigma;^2$ - дисперсия исследуемой метрики, $Z$ - величина критерия, зависящая от выбора $&alpha;$ и $&beta;$, $m$ - минимально детектируемый эффект (разница в метриках тестовой и контрольной групп).

### CUPED (Controlled-experiment using pre-experiment data)

Часто в экспериментах нас интересует разница метрик среднего в исследуемых группах и возникает потребность увеличить чувствительность A/B-теста, не оказывая влияния на значения метрики.

Метод CUPED применяется в том случае, когда исследуемая метрика - среднее. Данный метод позволяет снизить дисперсию метрики, сохранив ее значения неизменными. Покажем применение CUPED по шагам:

1. Пусть наши данные можно разделить следующим образом: X - данные до начала эксперимента, Y - данные после начала эксперимента (на которые эксперимент повлиял). Выберем такое X, которое независимо от эксперимента и скорреллированно с Y.
   
> Предлагается брать в качестве X исторические данные - как правило, это самый очевидный вариант, но необязательный. Главное о чем нужно помнить при выборе X - на него не должен влиять эксперимент, а также должна существовать корелляция между X и Y.
   
> На вопрос о том, за какой период брать данные для X теоретического ответа нет. Имеет смысл рассмотреть несколько вариантов и выбрать тот, где дисперсия наименьшая.

2. Напомним, что $\bar{X} = {{\sum}^n_{i=0} X_i \over n}$ согласно ЦПТ распределена нормально с параметрами $E(\bar{X}) = E(X)$ и $D(\bar{X}) = {D(X) \over n}$, где для всех i $D(X_i) = D(X)$ и $E(X_i) = E(X)$ - аналогично с $\bar{Y}$. Введем новую метрику:

```math 
\hat{Y} = \bar{Y} - &Theta;(\bar{X} - E(X))
```

> Запишем новую метрику более определенно: $\hat{Y_j} = \bar{Y_j} - &Theta;(\bar{X_j} - E(X_j))$, где $j$ - индекс группы в тесте.

  a. Покажем, что оценка математического ожидания $E(\bar{X})$ относительно $E(Y)$ является несмещенной, иначе говоря новая метрика $\hat{Y}$ совпадает по значениям со старой $\bar{Y}$.

```math 
E(\hat{Y}) = E(\bar{Y}) - &Theta;E(\bar{X}) + &Theta;E(E(X)) = \left(E(\bar{X}) = E(X); E(\bar{Y}) = E(Y) \right) = E(Y) - &Theta;E(X) + &Theta;E(X) = E(Y)
```

  b. Покажем, что дисперсия $D(\hat{Y})$ новой метрики является функцией $&Theta;$.

```math 
D(\hat{Y}) = \left(D(E(X)) = 0\right) = D(\bar{Y} - &Theta;\bar{X}) = \left(D(\bar{X}) = {D(X) \over n}; D(\bar{Y}) = {D(Y) \over n} \right) = {1 \over n} D(Y - &Theta;X) = {1 \over n} \left(D(Y) + &Theta;^2D(X) - 2&Theta;Cov(X,Y)\right)
```

> Бизнесу новую метрику можно представить как разницу в поведении пользователей до и после эксперимента.

3. Выберем $&Theta;$ таким, при котором $D(\hat{Y})$ принимает минимальное значение. Как видим, $D(\hat{Y}) = f(&Theta;)$ и дисперсия новой метрики квадратично зависит от $&Theta;$. Чтобы найти точку минимума, возьмем первую производную $f'(&Theta;)$ и приравняем ее к нулю ${1 \over n} \left(2&Theta;D(X) - 2Cov(X,Y) \right) = 0$. Дисперсия $D(\hat{Y})$ принимает минимальное значение при
   
 ```math 
&Theta;_0 = {Cov(X,Y) \over D(X)}
\Rightarrow
D_{min}(\hat{Y}) = {1 \over n} \left(D(Y) + {Cov^2(X,Y) \over D(X)} - 2{Cov^2(X,Y) \over D(X)}\right) = {D(Y) \over n} \left(1 - &rho;^2\right) = \left(D(\bar{Y}) = {D(Y) \over n}\right) = D(\bar{Y}) \left(1 - &rho;^2\right)
```

>  Обратите внимание, $&Theta;$ рассчитывается по всей выборке и принимает одно и только одно значение, которое далее применяется для каждой из групп.

Покажем теперь реализацию применения метода на практике на примере анализа гипотетического A/B-теста, предположительно влияющего на средний чек пользователя.

Импортируем необходимые библиотеки и считываем данные из файла. Структура данных: ```user_account_id``` - идентификатор пользователя, ```variation``` - группа эксперимента, ```avg_check_pre```/```avg_check_exp``` - значения среднего чека у пользователя до/после эксперимента.
 ```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FILE_NAME = 'exp_data_source_v7.xlsx'
EXP_DATA = 'avg_check_exp'
EXP_DATA_CUPED = 'avg_check_exp_cuped'
PRE_EXP_DATA = 'avg_check_pre'
VARIATION_A = 'A'
VARIATION_B = 'B'

exp_data = pd.read_excel(FILE_NAME)
```

Посмотрим на анализ теста без применения метода CUPED
 ```python
metric_A = np.mean(exp_data.loc[exp_data.variation==VARIATION_A, EXP_DATA])
metric_B = np.mean(exp_data.loc[exp_data.variation==VARIATION_B, EXP_DATA])

print(f'Variation {VARIATION_A}:', metric_A)
print(f'Variation {VARIATION_B}:', metric_B)
print(f'Div <Y({VARIATION_B})> - <Y({VARIATION_A})>:', metric_B - metric_A)
print(smf.ols(f'{EXP_DATA} ~ variation', data=exp_data).fit().summary().tables[1])
```

Определяем $&Theta;$ и коэффициент корреляции, рассчитываем новую метрику $\hat{Y}$
 ```python 
theta = smf.ols(f'{EXP_DATA} ~ {PRE_EXP_DATA}', data=exp_data).fit().params[1]
corr = exp_data[PRE_EXP_DATA].corr(exp_data[EXP_DATA])

print('\nTheta:', theta)
print('\nCoefcorr:', corr)

pre_exp_mean_A = np.mean(exp_data.loc[exp_data.variation==VARIATION_A, PRE_EXP_DATA])
pre_exp_mean_B = np.mean(exp_data.loc[exp_data.variation==VARIATION_B, PRE_EXP_DATA])

exp_data.loc[exp_data.variation==VARIATION_A, EXP_DATA_CUPED] = (
    exp_data[EXP_DATA] - theta * (exp_data[PRE_EXP_DATA] - pre_exp_mean_A)
)
exp_data.loc[exp_data.variation==VARIATION_B, EXP_DATA_CUPED] = (
    exp_data[EXP_DATA] - theta * (exp_data[PRE_EXP_DATA] - pre_exp_mean_B)
)
```

> Альтернативный способ нахождения $&Theta;$
> ```python 
> cov_X_Y = np.cov(exp_data[PRE_EXP_DATA], exp_data[EXP_DATA], ddof=1)[0, 1]
> var_X = np.var(exp_data[PRE_EXP_DATA], ddof=1)
> theta = cov_X_Y / var_X
> ```

Анализ теста с использованием метода CUPED
 ```python
metric_A_cuped = np.mean(exp_data.loc[exp_data.variation==VARIATION_A, EXP_DATA_CUPED])
metric_B_cuped = np.mean(exp_data.loc[exp_data.variation==VARIATION_B, EXP_DATA_CUPED])

print(f'Variation {VARIATION_A} (CUPED):', metric_A_cuped)
print(f'Variation {VARIATION_B} (CUPED):', metric_B_cuped)
print(f'Div <Y({VARIATION_B})> - <Y({VARIATION_A})> (CUPED):', metric_B_cuped - metric_A_cuped)
print(smf.ols(f'{EXP_DATA_CUPED} ~ variation', data=exp_data).fit().summary().tables[1])
```

Итоги применения метода CUPED:
- Значения метрик с применением/без применения метода CUPED не отличаются.
- Корреляция между X и Y оказалась достаточно хорошей, благодаря чему удалось значительно снизить дисперсию метрики.
- В обоих случаях обнаружена статистическая значимость изменений; можно утверждать, что для анализа данной метрики набранный объем данных избыточен.
