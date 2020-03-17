import pandas as pd
import arch
import numpy as np

from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator

from statsmodels.stats.diagnostic import het_breuschpagan


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
            s.model.fit_model()

    def training_summary(self):
        for s in self.stock_list:
            s.model.summary_fit()

    # def strat_from_model(self):
    #    for s in self.stock_list:
    #


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
        self.index_test = int(len(self.data) * 0.8)
        self.index_validation = int(self.index_test * 0.9)

        self.training_return, self.test_return = self.train_test_split()
        self.mean_train = None
        self.std_train = None
        self.normalisation()

        self.model = StockModel(stock=self,
                                return_model_type=return_model_type,
                                volatility_model_type=volatility_model_type)

    def read_data(self):
        df = pd.read_csv(self.data_path)
        df_small = df.iloc[1:42551]
        return df_small

    def train_test_split(self):
        return self.data.iloc[:self.index_test][self.return_column], \
               self.data.iloc[(self.index_test + 1):][self.return_column]

    def normalisation(self):
        self.mean_train = self.training_return.mean()
        self.std_train = self.training_return.std()
        self.training_return = (self.training_return - self.mean_train) / self.std_train
        self.test_return = (self.test_return - self.mean_train) / self.std_train

    def reshape_time_series(self,
                            dataset,
                            look_back=1):
        dataset = dataset.ravel()
        dataX = pd.DataFrame()
        dataY = pd.Series()
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back)]
            dataX = dataX.append([a])
            dataY = dataY.append(pd.Series(dataset[i + look_back]))
        return dataX, dataY

    def get_training_data(self):
        return self.training_return

    def get_test_data(self):
        return self.test_return

    def get_homoscedasticity_training_data(self):
        index_start = len(self.training_return) - len(self.model.heteroscedasticity)
        return self.training_return.iloc[index_start:] / self.model.heteroscedasticity.values

    def get_homoscedasticity_test_data(self):
        return self.test_return / self.model.heteroscedasticity

    def fit_model(self):
        self.model.fit_model()


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
        self.metrics_dict = dict()

        self.heteroscedasticity = None
        self.homoscedastic_return = None

    def fit_model(self):
        if self.return_model_type == 'AR' and self.volatility_model_type in ['arch', 'garch']:
            self.return_model = MixModel(stock_model=self,
                                         return_model_type=self.return_model_type,
                                         volatility_model_type=self.volatility_model_type)
            self.return_model.build_model()
            self.return_model.fit()

            self.return_model.evaluate()
            self.heteroscedasticity = self.return_model.fitted_conditional_variance
            self.homoscedastic_return = self.stock.get_homoscedasticity_training_data()

        else:
            return_model = ReturnModel(stock_model=self,
                                       model_type=self.return_model_type,
                                       return_train=self.data_training,
                                       index_validation=None,
                                       lags=1)
            return_model.build_model()
            return_model.fit()

            volatility_model = VolatilityModel(stock_model=self,
                                               model_type=self.volatility_model_type,
                                               resids=return_model.fitted_resids,
                                               index_validation=None,
                                               p=2)
            volatility_model.build_model()
            volatility_model.fit()

            self.heteroscedasticity = volatility_model.fitted_conditional_volatility
            self.homoscedastic_return = self.stock.get_homoscedasticity_training_data()
            self.return_model = ReturnModel(stock_model=self,
                                            model_type=self.return_model_type,
                                            return_train=self.homoscedastic_return,
                                            index_validation=self.stock.index_validation)
            self.return_model.build_model()
            self.return_model.fit()

            self.return_model.set_return_train(S=self.data_training)
            self.return_model.evaluate()

            self.volatility_model = VolatilityModel(stock_model=self,
                                                    model_type=self.volatility_model_type,
                                                    resids=self.return_model.all_resids,
                                                    index_validation=self.stock.index_validation)
            self.volatility_model.build_model()
            self.volatility_model.fit()

    def homoskedasticity_test(self):
        # bp_test = het_breuschpagan(resid=self.return_model.fitted_resids,
        #                            exog_het=self.return_model.r_true_train)
        # print(bp_test)

        self.metrics_dict['init_skewness'] = self.data_training.skew(axis=0)
        self.metrics_dict['init_kurtosis'] = self.data_training.kurtosis(axis=0)

        self.metrics_dict['final_skewness'] = self.homoscedastic_return.skew(axis=0)
        self.metrics_dict['init_kurtosis'] = self.homoscedastic_return.kurtosis(axis=0)

    def evaluate(self):
        self.return_model.evaluate()
        self.metrics_dict['mse_train'] = self.mse(r_pred=self.return_model.r_pred_train,
                                                  r_true=self.return_model.r_true_train)
        self.metrics_dict['mse_validation'] = self.mse(r_pred=self.return_model.r_pred_validation,
                                                       r_true=self.return_model.r_true_validation)

        self.volatility_model.evaluate()
        self.metrics_dict['accuracy_train'] = self.accuracy(r_pred=self.return_model.get_prediction_train(),
                                                            r_true=self.return_model.r_true_train,
                                                            vol_pred=self.volatility_model.get_prediction_train())

        self.metrics_dict['accuracy_validation'] = self.accuracy(r_pred=self.return_model.get_prediction_validation(),
                                                                 r_true=self.return_model.r_true_validation,
                                                                 vol_pred=self.volatility_model.get_prediction_validation())

    @staticmethod
    def mse(r_pred,
            r_true):
        merged_df = r_pred.merge(right=r_true, left_index=True, right_index=True, how='inner')
        columns = merged_df.columns
        return mean_squared_error(y_true=merged_df[columns[0]],
                                  y_pred=merged_df[columns[1]])

    @staticmethod
    def accuracy(r_pred,
                 r_true,
                 vol_pred,
                 quantile=1.96):
        accuracy = []
        imin = max(min(r_pred.index), min(r_true.index), min(vol_pred.index))
        imax = min(max(r_pred.index), max(r_true.index), max(vol_pred.index))
        for i in range(imin, imax+1):
            mu = r_pred.loc[i]
            sigma = abs(vol_pred.loc[i])
            accuracy.append(int((r_true.loc[i] > mu-quantile*sigma) & (r_true.loc[i] < mu+quantile*sigma)))

        return np.mean(accuracy)

    def summary_fit(self):
        self.return_model.summary()
        print(self.metrics_dict)


class MixModel:
    def __init__(self,
                 stock_model,
                 return_model_type,
                 volatility_model_type,
                 lags=4):

        self.stock_model = stock_model
        self.return_model_type = return_model_type
        self.volatility_model_type = volatility_model_type
        self.index_validation = self.stock_model.stock.index_validation

        self.model = None
        self.model_dict = None
        self.lags = lags

        self.fitted_model = None
        self.fitted_resids = None
        self.fitted_conditional_variance = None

        self.r_pred_train = None
        self.r_pred_validation = None
        self.vol_pred_train = None
        self.vol_pred_validation = None

        self.r_true_train = self.stock_model.data_training[self.lags:self.index_validation]
        self.r_true_validation = self.stock_model.data_training[self.index_validation + self.lags + 1:]

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

    def summary(self):
        self.fitted_model.summary()

    def evaluate(self):
        forecast = self.fitted_model.forecast(horizon=1)
        forecast_return = forecast.mean.dropna()
        self.r_pred_train = forecast_return.iloc[:self.index_validation]
        self.r_pred_validation = forecast_return.iloc[self.index_validation + 1:]

        forecast_volatility = forecast.variance.dropna()
        self.vol_pred_train = forecast_volatility.iloc[:self.index_validation]
        self.vol_pred_validation = forecast_volatility.iloc[self.index_validation + 1:]

        self.fitted_conditional_variance = pd.concat([self.vol_pred_train, self.vol_pred_validation], axis=0)


class ReturnModel:
    def __init__(self,
                 stock_model,
                 model_type,
                 return_train,
                 index_validation=None,
                 lags=5):
        self.stock_model = stock_model
        self.model_type = model_type
        self.return_train = return_train
        if index_validation is None:
            self.index_validation = len(self.return_train) - 1
        else:
            self.index_validation = index_validation - self.return_train.index[0]

        self.generator_train = None
        self.model = None
        self.model_dict = None
        self.lags = lags

        self.fitted_model = None
        self.fitted_resids = None
        self.all_resids = None

        self.overwrite_params = None

        self.r_pred_train = None
        self.r_pred_validation = None
        self.r_true_train = self.return_train[self.lags:self.index_validation]
        self.r_true_validation = self.return_train[self.index_validation + self.lags + 1:]

    def build_model(self):
        if self.model_type == 'AR':
            self.model_dict = {'mean': self.model_type,
                               'lags': self.lags}
            self.model = arch.arch_model(y=self.return_train.ravel(),
                                         rescale=False,
                                         **self.model_dict)

        elif self.model_type == 'NN':
            self.model_dict = {'layers': [10, 1]}
            self.model = Sequential()
            self.model.add(Dense(units=self.model_dict['layers'][0],
                                 activation='sigmoid',
                                 input_dim=self.lags))
            #self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.model_dict['layers'][1],
                                 activation='linear'))
            self.model.compile(loss="mse",
                               optimizer="adam")

        elif self.model_type == 'RNN':
            self.model_dict = {'layers': [10, 1]}
            self.model = Sequential()

            self.model.add(SimpleRNN(
                units=self.model_dict['layers'][0],
                activation='tanh',
                use_bias=True,
                dropout=0.2,
                return_sequences=False))

            self.model.add(Dense(
                output_dim=self.model_dict['layers'][1]))
            self.model.add(Activation("linear"))
            self.model.compile(loss="mse",
                               optimizer="rmsprop")

    def fit(self):
        if self.model_type == 'AR':
            self.fitted_model = self.model.fit(update_freq=5,
                                               last_obs=self.index_validation)

            self.fitted_resids = pd.Series(self.model.resids(
                params=self.fitted_model.params[:(self.model_dict['lags'] + 1)]))

        elif self.model_type == 'NN':
            self.generator_train = TimeseriesGenerator(data=self.return_train.ravel(),
                                                       targets=self.return_train.ravel(),
                                                       length=self.lags,
                                                       batch_size=8,
                                                       end_index=self.index_validation)
            self.model.fit_generator(self.generator_train,
                                     steps_per_epoch=1,
                                     epochs=200,
                                     verbose=0,
                                     use_multiprocessing=False)
            self.fitted_model = self.model

            y_pred_train = self.fitted_model.predict_generator(self.generator_train, verbose=0)

            self.fitted_resids = y_pred_train[:, 0] - self.return_train.iloc[self.lags:self.index_validation]

    def evaluate(self):
        if self.model_type == 'AR':
            if self.overwrite_params is None:
                params = self.fitted_model.params
            else:
                params = self.overwrite_params
            forecast = self.fitted_model.forecast(horizon=1,
                                                  params=params,
                                                  start=self.lags) \
                .mean.dropna()
            self.r_pred_train = forecast.iloc[:self.index_validation]
            self.r_pred_validation = forecast.iloc[self.index_validation+1:]

        elif self.model_type == 'NN':
            self.r_pred_train = pd.Series(self.fitted_model.predict_generator(self.generator_train)[:, 0],
                                          index=self.return_train.index[self.lags:self.index_validation])

            validation_generator = TimeseriesGenerator(data=self.return_train.ravel(),
                                                       targets=self.return_train.ravel(),
                                                       length=self.lags,
                                                       start_index=self.index_validation)
            self.r_pred_validation = pd.Series(self.fitted_model.predict_generator(validation_generator)[:, 0],
                                               index=self.return_train.index[self.index_validation+self.lags+1:])

        self.all_resids = pd.Series(pd.concat([self.r_pred_train, self.r_pred_validation], axis=0).values[:, 0])

    def get_prediction_train(self):
        return self.r_pred_train

    def get_prediction_validation(self):
        return self.r_pred_validation

    def summary(self):
        self.fitted_model.summary()

    def set_return_train(self,
                         S):
        self.return_train = S
        self.r_true_train = self.return_train[self.lags:self.index_validation]
        self.r_true_validation = self.return_train[self.index_validation + self.lags + 1:]
        if self.model_type == 'AR':
            self.overwrite_params = self.fitted_model.params
            self.build_model()
            self.fit()
        elif self.model_type == 'NN':
            self.generator_train = TimeseriesGenerator(data=self.return_train.ravel(),
                                                       targets=self.return_train.ravel(),
                                                       length=self.lags,
                                                       batch_size=8,
                                                       end_index=self.index_validation)


class VolatilityModel:
    def __init__(self,
                 stock_model,
                 model_type,
                 resids,
                 index_validation=None,
                 p=1,
                 q=1):

        self.stock_model = stock_model
        self.model_type = model_type
        self.resids = resids

        self.p = p
        self.q = q

        if index_validation is None:
            self.index_validation = len(self.resids) - 1
        else:
            self.index_validation = index_validation - self.resids.index[0]

        self.generator_train = None
        self.model = None
        self.model_dict = None
        self.fitted_model = None

        self.fitted_conditional_volatility = None

        self.vol_pred_train = None
        self.vol_pred_validation = None
        self.vol_true_train = None
        self.vol_true_validation = None

    def build_model(self):
        if self.model_type in ['arch', 'garch']:
            self.model_dict = {'mean': 'zero',
                               'vol': self.model_type}
            if self.model_type == 'arch':
                self.model_dict['p'] = self.p
            elif self.model_type == 'garch':
                self.model_dict['p'] = self.p
                self.model_dict['q'] = self.q
            self.model = arch.arch_model(y=self.resids.ravel(),
                                         rescale=False,
                                         **self.model_dict)
        elif self.model_type == 'NN':
            self.model_dict = {'layers': [10, 1],
                               'p': self.p}
            self.model = Sequential()
            self.model.add(Dense(units=self.model_dict['layers'][0],
                                 activation='sigmoid',
                                 input_dim=self.model_dict['p']))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.model_dict['layers'][1],
                                 activation='linear'))
            self.model.compile(loss="mse",
                               optimizer="adam")

    def fit(self):
        self.vol_true_train = self.resids.iloc[self.model_dict['p']:self.index_validation]
        self.vol_true_validation = self.resids.iloc[self.index_validation + self.model_dict['p'] + 1:]

        if self.model_type in ['arch', 'garch']:
            self.fitted_model = self.model.fit(update_freq=5)

            self.fitted_conditional_volatility = self.fitted_model.conditional_volatility.dropna().apply(
                lambda var: var ** 0.5)

        elif self.model_type == 'NN':
            self.generator_train = TimeseriesGenerator(data=self.resids.apply(lambda resid: resid ** 2).values,
                                                       targets=self.resids.apply(lambda resid: resid ** 2).values,
                                                       length=self.model_dict['p'],
                                                       batch_size=8,
                                                       end_index=self.index_validation)
            self.model.fit_generator(self.generator_train,
                                     steps_per_epoch=1,
                                     epochs=200,
                                     verbose=0,
                                     use_multiprocessing=False)
            self.fitted_model = self.model

            y_pred = self.fitted_model.predict_generator(self.generator_train, verbose=0)
            # TODO: Pour l'instant les prédictions de variances peuevent être négatives ....
            self.fitted_conditional_volatility = pd.Series(y_pred[:, 0]).apply(lambda var: abs(var) ** 0.5)

    def evaluate(self):
        if self.model_type == 'AR':
            self.vol_pred_train = self.fitted_model.forecast(end=self.index_validation,
                                                             horizon=1) \
                .volatility.dropna()
            self.vol_pred_validation = self.fitted_model.forecast(start=self.index_validation + 1,
                                                                  horizon=1) \
                .volatility.dropna()

        elif self.model_type == 'NN':
            self.vol_pred_train = pd.Series(self.fitted_model.predict_generator(self.generator_train)[:, 0],
                                            index=self.resids.index[self.model_dict['p']:self.index_validation])

            validation_generator = TimeseriesGenerator(data=self.resids.ravel(),
                                                       targets=self.resids.ravel(),
                                                       length=self.model_dict['p'],
                                                       start_index=self.index_validation)
            self.vol_pred_validation = pd.Series(self.fitted_model.predict_generator(validation_generator)[:, 0],
                                                 index=self.resids.index[
                                                       self.index_validation + self.model_dict['p'] + 1:])

    def get_prediction_train(self):
        return self.vol_pred_train

    def get_prediction_validation(self):
        return self.vol_pred_validation

    def summary(self):
        self.fitted_model.summary()


if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    data_path = 'clean_data.csv'
    return_model_type = 'AR'
    volatility_model_type = 'NN'
    return_column = 'log_return'

    stock = Stock(stock_name='Bnp Paribas',
                  data_path=data_path,
                  return_column=return_column,
                  return_model_type=return_model_type,
                  volatility_model_type=volatility_model_type)

    stock.fit_model()

    stock.model.evaluate()

    stock.model.homoskedasticity_test()

    stock.model.summary_fit()

    # portfolio = Portfolio(stocks_names=['Bnp Paribas'],
    #                       stocks_paths=[data_path],
    #                       return_column=return_column,
    #                       return_model_types=[return_model_type],
    #                       volatility_model_types=[volatility_model_type])
    #
    # portfolio.model_training()

    # model = arch.arch_model(y=df_small['log_return'], mean='AR', lags=2, vol='arch', p=5)

    # res = model.fit(update_freq=5, last_obs=index_train)
    # print(res.summary())
    # res.plot(annualize='D')

    # pred = res.forecast(start=index_train + 1, horizon=3)
    # pred.mean.dropna().head(10)

