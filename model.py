import pandas as pd
import arch

from statsmodels.tsa.ar_model import AR
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error

df = pd.read_csv('clean_data.csv')

# Data analysis
# Mean, variance, skewness, kurtosis

# Correlograme
#plot_acf()

n = len(df)
n_train = int(n*0.7)
#train = df.iloc[:n_train]
#test = df.iloc[n_train:]

df_small = df.iloc[1:10000]
#df_small.set_index('date', inplace=True)

index_train = 7000
model = arch.arch_model(y=df_small['log_return'], mean='AR', lags=2, vol='arch', p=5, rescale=True)
# TODO: rescale
res = model.fit(update_freq=5, last_obs=index_train)
print(res.summary())
res.plot(annualize='D')

pred = res.forecast(start=index_train+1,
                    horizon=3)
pred.mean.dropna().head(10)


def return_model(return_model_type,
                 train):

    if return_model_type == 'AR':
        lag = 5
        model = arch.univariate.ARX(train['log_return'], lags=lag,  rescale=True)
        model_result = model.fit(train['log_return'], disp='off', show_warning=False)

        #print('Model order: %s' % model.k_ar)
        #print('Coefficients: %s:' % model.params)
        model_result.summary()

        model_forecast = model_result.forecast(start=lag+1)
        return_mean = model_forecast.mean
        return_variance = model_forecast.variance

    return model


def volatility_model(volatility_model_type,
                     train_squared_error):

    if volatility_model_type == 'ARCH':
        model = AR(train_squared_error).fit()

        print('Model order: %s' % model.k_ar)
        print('Coefficients: %s:' % model.params)

    return model


def model_pipeline(train,
                   test,
                   return_model_type='AR',
                   volatility_model_type='ARCH'):

    model_return1 = return_model(return_model_type=return_model_type,
                                 train=train)

    predicted_return1 = model_return1.forecast(train)
    p1 = model_return1.k_ar
    return_squared_error = (train.iloc[p1:] - predicted_return1) ** 2

    model_volatility1 = volatility_model(volatility_model_type=volatility_model_type,
                                         train_squared_error=return_squared_error)
    q1 = model_volatility1.k_ar
    heteroscedasticity = model_volatility1.forecast(return_squared_error)

    homoscedasticity_train = train.iloc[q1:].div(heteroscedasticity)

    model_return2 = return_model(return_model_type=return_model_type,
                                 train=homoscedasticity_train)
    p2 = model_return2.k_ar
    predicted_return2 = model_return1.forecast(test)
    return_squared_error2 = (test.iloc[p2:] - predicted_return2) ** 2

    model_volatility2 = volatility_model(volatility_model_type=volatility_model_type,
                                         train_squared_error=return_squared_error2)

    volatitlity_test = model_volatility2.forecast(return_squared_error2)
