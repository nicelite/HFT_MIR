import pandas as pd

from statsmodels.tsa.ar_model import AR
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error

df = pd.read_excel('Bnp Paribas.xlsx', header = None)
df.columns = ['Volume', 'Price', 'datetime']

# Data analysis
# Mean, variance, skewness, kurtosis

# Correlograme
#plot_acf()


def return_model(return_model_type,
                 train):

    if return_model_type == 'AR':
        model = AR(train).fit()

        print('Model order: %s' % model.k_ar)
        print('Coefficients: %s:' % model.params)

    return model


def volatility_model(return_model_type,
                     train_squared_error):

    if return_model_type == 'AR':
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

    predicted_return1 = model_return1.predict(train)
    return_squared_error = (train - predicted_return1) ** 2

    model_volatility1 = model


