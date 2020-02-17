import pandas as pd
import arch

from sklearn.metrics import mean_squared_error


class StockModel:
    def __init__(self,
                 stock,
                 return_model_type='AR',
                 volatility_model_type='arch'):

        self.return_model_type = return_model_type
        self.volatility_model_type = volatility_model_type
        self.stock = stock
        self.data_training = self.stock.get_training_data()

        self.return_model = None
        self.volatility_model = None
        self.model_dict = None

    def fit_model(self):
        if self.return_model_type == 'AR' and self.volatility_model_type in ['arch', 'garch']:
            self.return_model = MixModel(stock_model=self,
                                         return_model_type=self.return_model_type,
                                         volatility_model_type=self.volatility_model_type)
            self.return_model.fit()

        else:
            self.return_model = ReturnModel(stock_model=self,
                                            model_type=self.return_model_type)
            self.return_model.fit()

            self.volatility_model = VolatilityModel(stock_model=self,
                                                    model_type=self.volatility_model_type,
                                                    resids=self.return_model.fitted_resids)
            self.volatility_model.fit()

    def evaluate_model(self):

        if self.return_model_type == 'AR':
            model_forecast = self.fitted_return_model.forecast(horizon=1)

            forecast_return = model_forecast.mean.dropna()
            forecast_volatility = model_forecast.variance.dropna()

            forecast_return_train = forecast_return.iloc[:self.stock.index_validation]
            # forecast_volatility_train = forecast_volatility.iloc[:self.stock.index_validation]

            forecast_return_validation = forecast_return.iloc[(self.stock.index_validation + 1):]
            # forecast_volatility_test = forecast_volatility.iloc[(self.stock.index_validation + 1):]

            self.mse_return_train = mean_squared_error(y_true=self.data_training.iloc[:self.stock.index_validation],
                                                       y_pred=forecast_return_train)

            self.mse_return_validation = mean_squared_error(
                y_true=self.data_training.iloc[(self.stock.index_validation + 1):],
                y_pred=forecast_return_validation)


class ReturnModel:
    def __init__(self,
                 stock_model,
                 model_type,
                 lags=5):
        self.stock_model = stock_model
        self.model_type = model_type
        self.model = None
        self.model_dict = None
        self.lags = lags

        self.fitted_model = None
        self.fitted_resids = None

    def build_model(self):
        self.model_dict = {'mean': self.model_type,
                           'lags': self.lags}
        self.model = arch.arch_model(y=self.stock_model.data_training,
                                     rescale=False,
                                     **self.model_dict)

    def fit(self):
        if self.model_type == 'AR':
            self.fitted_model = self.model.fit(update_freq=5,
                                               last_obs=self.stock_model.stock.index_validation)

            self.fitted_resids = pd.Series(self.model.resids(
                params=self.fitted_model.params[:(self.model_dict['lags'] + 1)]))


class MixModel:
    def __init__(self,
                 stock_model,
                 return_model_type,
                 volatility_model_type,
                 lags=5):

        self.stock_model = stock_model
        self.return_model_type = return_model_type
        self.volatility_model_type = volatility_model_type
        self.model = None
        self.model_dict = None
        self.lags = lags

        self.fitted_model = None
        self.fitted_resids = None

    def build_model(self):
        self.model_dict = {'mean': self.return_model_type,
                           'lags': self.lags,
                           'vol': self.volatility_model_type}
        if self.volatility_model_type == 'arch':
            self.model_dict['p'] = 1
        elif self.volatility_model_type == 'garch':
            self.model_dict['p'] = 1
            self.model_dict['q'] = 1
        self.model = arch.arch_model(y=self.stock_model.data_training,
                                     rescale=False,
                                     **self.model_dict)

    def fit(self):
        self.fitted_model = self.model.fit(update_freq=5,
                                           last_obs=self.stock_model.stock.index_validation)


class VolatilityModel:
    def __init__(self,
                 stock_model,
                 model_type,
                 resids):

        self.stock_model = stock_model
        self.model_type = model_type
        self.resids = resids

        self.model = None
        self.model_dict = None
        self.fitted_model = None

    def build_model(self):
        if self.model_type in ['arch', 'garch']:
            self.model_dict = {'mean': 'zero',
                               'vol': self.model_type}
            if self.model_type == 'arch':
                self.model_dict['p'] = 1
            elif self.model_type == 'garch':
                self.model_dict['p'] = 1
                self.model_dict['q'] = 1
            self.model = arch.arch_model(y=self.resids,
                                         rescale=False,
                                         **self.model_dict)

    def fit(self):
        self.fitted_model = self.model.fit(update_freq=5)


class Stock:
    def __init__(self,
                 data_path='clean_data.csv',
                 stock_name='Bnp Paribas',
                 return_column='log_return',
                 return_model_type='AR',
                 volatility_model_type='arch'):

        self.data_path = data_path
        self.stock_name = stock_name
        self.data = self.read_data()
        self.return_column = return_column
        self.index_test = int(len(self.data)*0.8)
        self.index_validation = int(self.index_test*0.9)

        self.training_return, self.test_return = self.train_test_split()
        self.mean_train = None
        self.std_train = None
        self.normalisation()

        self.model = StockModel(stock=self,
                                return_model_type=return_model_type,
                                volatility_model_type=volatility_model_type)

    def read_data(self):
        df = pd.read_csv(self.data_path)
        df_small = df.iloc[1:10000]
        return df_small

    def train_test_split(self):
        return self.data.iloc[:self.index_test][self.return_column], \
               self.data.iloc[(self.index_test+1):][self.return_column]

    def normalisation(self):
        self.mean_train = self.training_return.mean()
        self.std_train = self.training_return.std()
        self.training_return = (self.training_return - self.mean_train) / self.std_train
        self.test_return = (self.test_return - self.mean_train) / self.std_train

    def get_training_data(self):
        return self.training_return

    def get_test_data(self):
        return self.test_return

    def fit_model(self):
        self.model.fit()


class Portfolio:
    def __init__(self,
                 stocks_names,
                 stocks_paths,
                 return_column,
                 return_model_types,
                 volatility_model_types):
        self.stock_list = list()
        for i in range(len(stocks_names)):
            self.stock_list.append(Stock(stock_name=stocks_names[i],
                                         data_path=stocks_paths[i],
                                         return_column=return_column,
                                         return_model_type=return_model_types[i],
                                         volatility_model_type=volatility_model_types[i]))

    def model_training(self):
        for s in self.stock_list:
            s.model.fit()

    #def strat_from_model(self):
    #    for s in self.stock_list:
    #


if __name__ == '__main__':

    pd.set_option('mode.chained_assignment', None)
    data_path = 'clean_data.csv'
    return_model_type = 'AR'
    volatility_model_type = 'arch'
    return_column = 'log_return'

    portfolio = Portfolio(stocks_names=['Bnp Paribas'],
                          stocks_paths=[data_path],
                          return_column=return_column,
                          return_model_types=[return_model_type],
                          volatility_model_types=[volatility_model_type])

    portfolio.model_training()

    #model = arch.arch_model(y=df_small['log_return'], mean='AR', lags=2, vol='arch', p=5)

    #res = model.fit(update_freq=5, last_obs=index_train)
    #print(res.summary())
    #res.plot(annualize='D')

    #pred = res.forecast(start=index_train + 1, horizon=3)
    #pred.mean.dropna().head(10)