import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from workalendar.europe import Russia
import os
import pickle
from pprint import pprint

FILENAME = 'hackaton2023_train.gzip'
PATH = os.getcwd()
CLASSTERS_FILE = './classters.pkl'
TRAIN = True
DEBUG = True

data = pd.read_parquet(PATH + "./" + FILENAME).iloc[:1000]

data.drop('group_name', axis=1, inplace=True)
# Кластеризация товаров чека
with open(PATH + CLASSTERS_FILE, 'rb') as f:
    classters = pickle.load(f)

def find_classter_name(words):
    for key in sorted(classters.keys()):
        for word in words.split():
            if word.strip().lower() in classters[key]:
                return key
    else: return '_other'

data['dish_name'] = data['dish_name'].map(find_classter_name)

indexes = data['dish_name'].astype('category').cat.codes.to_numpy()
values = np.zeros((len(data), len(classters.keys())))

for _data, i in zip(values, indexes):
    _data[i] += 1
data[sorted(list(classters.keys()))] = values.astype(np.int8)
del values, indexes

# группировка по чекам
aggregated_data_cheque = data.groupby(['customer_id', 'startdatetime'], as_index=False).agg({
    'revenue': 'sum',
    'dish_name': ', '.join,
    'format_name': 'first', 
    'ownareaall_sqm': 'first', 
    **{key: 'first' for key in classters.keys()}}.update((
        {'buy_post': 'first', 'date_diff_post': 'first', } if TRAIN else {})))

if DEBUG:
    params = {
    'revenue': 'sum',
    'dish_name': ', '.join,
    'format_name': 'first', 
    'ownareaall_sqm': 'first', 
    **{key: 'first' for key in classters.keys()}}.update((
        {'buy_post': 'first', 'date_diff_post': 'first', } if TRAIN else {}))
    assert params.get('buy_post')
    assert params.get('date_diff_post')
    for key in classters.keys():
        assert params.get(key)

# убираем самые большие и маленькие чеки
reve_column = aggregated_data_cheque['revenue']
Q1 = reve_column.quantile(0.25)
Q3 = reve_column.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
aggregated_data_cheque = aggregated_data_cheque[(reve_column >= lower_bound) & (reve_column <= upper_bound)]

# временные фичи
def time_of_day(hour):
    if 0 <= hour < 6:
        return 'Ночь'
    elif 6 <= hour < 12:
        return 'Утро'
    elif 12 <= hour < 18:
        return 'День'
    else:
        return 'Вечер'
    
def time_of_day_h(hour):
    if 0 <= hour < 6:
        return '0'
    elif 6 <= hour < 12:
        return '1'
    elif 12 <= hour < 18:
        return '2'
    else:
        return '3' 
    
# бизнес
aggregated_data_cheque['date'] = aggregated_data_cheque['startdatetime'].dt.date
aggregated_data_cheque['time'] = aggregated_data_cheque['startdatetime'].dt.time
aggregated_data_cheque['year'] = aggregated_data_cheque['startdatetime'].dt.year


#модель (+aggregated_data_cheque['year'], но без aggregated_data_cheque['Year-month'])
aggregated_data_cheque['day_h'] = aggregated_data_cheque['startdatetime'].dt.dayofweek
aggregated_data_cheque['time_h'] = aggregated_data_cheque['time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
aggregated_data_cheque['time_name_h'] = aggregated_data_cheque['time'].apply(lambda x: time_of_day_h(x.hour))
aggregated_data_cheque['month_h'] = aggregated_data_cheque['startdatetime'].dt.month
aggregated_data_cheque['Year-month'] = aggregated_data_cheque['year'].astype(str) + '-' + aggregated_data_cheque['month_h'].astype(str)
aggregated_data_cheque['weekday_h'] = (aggregated_data_cheque['startdatetime'].dt.dayofweek >= 5).astype(int)
aggregated_data_cheque['quarter_h'] = aggregated_data_cheque['startdatetime'].dt.quarter

cal = Russia()
aggregated_data_cheque['date'] = pd.to_datetime(aggregated_data_cheque['date'])
aggregated_data_cheque['is_holiday'] = aggregated_data_cheque['date'].apply(lambda x: cal.is_holiday(x)).astype(int)

# Группируем данные по дате и рассчитываем средний чек
average_check_by_date = aggregated_data_cheque.groupby('date')['revenue'].median()
aggregated_data_cheque = pd.merge(aggregated_data_cheque, average_check_by_date, on='date', how='inner')

aggregated_data_cheque = aggregated_data_cheque.rename(columns={'revenue_x': 'revenue','revenue_y': 'revenue_day'})

# Группируем данные по пользователю и рассчитываем средний чек в месяц
average_check_by_rev = aggregated_data_cheque.groupby('customer_id')['revenue'].median()
aggregated_data_cheque = pd.merge(aggregated_data_cheque, average_check_by_rev, on='customer_id', how='inner')

aggregated_data_cheque = aggregated_data_cheque.rename(columns={'revenue_x': 'revenue','revenue_y': 'revenue_month_custom'})

aggregated_data_cheque['delta_mean_day'] = aggregated_data_cheque['revenue_day']-aggregated_data_cheque['revenue']

grouped_data = aggregated_data_cheque.groupby('customer_id',as_index=False)

trend = []

# определение тенденции по чекам у пользователя
model = LinearRegression()
grouped_data = aggregated_data_cheque.groupby('customer_id', as_index=False)
for customer_id, group in grouped_data:
    X = group[['startdatetime']].apply(lambda x: x.values.astype(int)).values.reshape(-1, 1)
    y = group['revenue']
    model.fit(X, y)
    trend_coefficient = model.coef_[0]
    trend.append({'customer_id': customer_id, 'coef_trend': trend_coefficient})
results_df = pd.DataFrame(trend)

aggregated_data_cheque = pd.merge(aggregated_data_cheque, results_df, on='customer_id', how='inner')

# время и покупки вместе
df = aggregated_data_cheque.sort_values(by=['customer_id', 'startdatetime'])
df['time_diff'] = df.groupby('customer_id')['startdatetime'].diff()
purchase_count = df.groupby('customer_id').size()
total_time_between_purchases = df.groupby('customer_id')['time_diff'].sum()
average_time_between_purchases = total_time_between_purchases / (purchase_count - 1)

result_df = pd.DataFrame({
    'customer_id': purchase_count.index,
    'purchase_count': purchase_count.values,
    'total_time_between_purchases': total_time_between_purchases.values,
    'average_time_between_purchases': average_time_between_purchases.values
})

aggregated_data_cheque = pd.merge(aggregated_data_cheque, result_df, on='customer_id', how='inner')

aggregated_data_cheque['min_customer'] = aggregated_data_cheque['revenue']
aggregated_data_cheque['max_customer'] = aggregated_data_cheque['revenue']
aggregated_data_cheque['std_customer'] = aggregated_data_cheque['revenue']

aggregated_data_cheque['average_time_between_purchases'] = aggregated_data_cheque['average_time_between_purchases'].dt.total_seconds()
aggregated_data_cheque['total_time_between_purchases'] = aggregated_data_cheque['total_time_between_purchases'].dt.total_seconds()

def mode_func(x):
    modes = x.mode()
    return modes.iloc[0] if not modes.empty else None

aggregated_data_user = aggregated_data_cheque

# aggregated_data_user['revenue_delta_class'] = aggregated_data_user.apply(lambda row: 495.0359785392551 - row['revenue'] if abs(row['delta_mean_day'])> 43 else 491.0214675593032 - row['revenue'], axis=1)

# признаки места 
aggregated_data_user['toilet']= aggregated_data_user['format_name'].apply(lambda x: 1 if 'с туалетом' in x else 0)
aggregated_data_user['Free-standing']=aggregated_data_user['format_name'].apply(lambda x: 1 if 'Отдельно стоящий' in x else 0)
aggregated_data_user['external zone']=aggregated_data_user['format_name'].apply(lambda x: 1 if 'без внешней зоны' in x else 0)

# продуктовые метрики 
patern = pd.DataFrame()
patern['customer_id'] = aggregated_data_user['customer_id']
patern['dish_name'] = aggregated_data_user['dish_name']
patern['most_common_pattern'] = patern['dish_name'].apply(lambda x: pd.Series(x.split(',')).mode().iat[0])
patern['pattern_count'] = patern['dish_name'].apply(lambda x: pd.Series(x.split(',')).value_counts().iat[0])

patern = patern.drop(columns='dish_name')

aggregated_data_user = pd.merge(aggregated_data_user, patern, on='customer_id', how='inner')

aggregated_data_user['total number of positions']= aggregated_data_user['dish_name'].str.count(',')+1

aggregated_data_user.to_parquet('END.parquet')
