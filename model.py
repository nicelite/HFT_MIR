from numpy.random import seed
from tensorflow import set_random_seed
import warnings
import pandas as pd

import arch
import numpy as np
from math import sqrt
import time

from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import TimeseriesGenerator

warnings.filterwarnings('ignore',
                        category=FutureWarning)

class Portfolio:
    def __init__(self,
                 stocks_names,
                 stocks_paths,
                 return_column,
                 model_dicts):
        self.stock_list = list()
        for i in range(len(stocks_names)):
            self.stock_list.append(Stock(stock_name=stocks_names[i],
                                         data_path=stocks_paths[i],
                                         return_column=return_column,
                                         model_dict=model_dicts[i]))

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
                 model_dict,
                 data_path='clean_data.csv',
                 stock_name='Bnp Paribas',
                 return_column='log_return'):
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
                                model_dict=model_dict)

    def read_data(self):
        df = pd.read_csv(self.data_path)
        df_small = df.iloc[1:42551]
        return df_small

    def train_test_split(self):
        return self.data.iloc[:self.index_test][self.return_column], \
               self.data.iloc[(self.index_test + 1):][self.return_column]

    def normalisation(self):
        self.mean_train = self.training_return.mean()
        #self.mean_train = 0
        self.std_train = self.training_return.std()
        #self.std_train = 1
        self.training_return = (self.training_return - self.mean_train) / self.std_train
        self.test_return = (self.test_return - self.mean_train) / self.std_train

    def remove_normalisation(self,
                             series,
                             type='return'):
        if series is not None:
            if type == 'return':
                return series * self.std_train + self.mean_train
            else:
                return series * self.std_train
        return None

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

    def get_homoscedasticity_endo_feature(self):
        hetero_values = self.model.heteroscedasticity.values
        if len(hetero_values.shape) > 1:
            hetero_values = hetero_values[:, 0]
        return hetero_values

    def get_homoscedasticity_training_data(self,
                                           heteroscedasticity=None):
        if heteroscedasticity is None:
            heteroscedasticity = self.model.heteroscedasticity
        index_start = len(self.training_return) - len(heteroscedasticity)
        hetero_values = heteroscedasticity.values
        if len(hetero_values.shape) > 1:
            hetero_values = hetero_values[:, 0]
        return self.training_return.iloc[index_start:] / hetero_values

    def get_homoscedasticity_test_data(self):
        hetero_values = self.model.heteroscedasticity.values
        if len(hetero_values.shape) > 1:
            hetero_values = hetero_values[:, 0]
        return self.test_return / hetero_values

    def fit_model(self):
        self.model.fit_model()

    def set_model_attribute(self,
                            model_type,
                            attribute,
                            value):
        self.model.set_model_attribute(model_type=model_type,
                                       attribute=attribute,
                                       value=value)

    def get_metrics(self):
        return self.model.metrics_dict

    def reset_metrics(self):
        self.model.reset_metrics()


class StockModel:
    def __init__(self,
                 stock,
                 model_dict):

        self.model_dict = model_dict
        self.return_model_type = model_dict['return_model']['type']
        self.volatility_model_type = model_dict['volatility_model']['type']
        self.stock = stock
        self.data_training = self.stock.get_training_data()

        self.return_model = None
        self.volatility_model = None
        self.metrics_dict = dict()

        self.heteroscedasticity = None
        self.homoscedastic_return = None

    def fit_iteration(self,
                      prev_return_model,
                      prev_volatility_model,
                      step=1):
        if step == self.model_dict['stock_model']['n_iteration']:
            index_validation = self.stock.index_validation
        else:
            index_validation = None

        self.evaluate(return_model=prev_return_model,
                      volatility_model=prev_volatility_model,
                      evaluate_validation=False,
                      step=step)

        self.heteroscedasticity = prev_volatility_model.fitted_conditional_volatility
        self.homoscedastic_return = self.stock.get_homoscedasticity_training_data()

        self.return_model = ReturnModel(stock_model=self,
                                        model_dict=self.model_dict['return_model'],
                                        return_train=self.homoscedastic_return,
                                        index_validation=index_validation,
                                        endogenous_feature=self.stock.get_homoscedasticity_endo_feature())
        self.return_model.build_model()
        self.return_model.fit()

        self.return_model.set_return_train(S=self.data_training)
        self.evaluate_return_model()

        self.volatility_model = VolatilityModel(stock_model=self,
                                                model_dict=self.model_dict['volatility_model'],
                                                resids=self.return_model.all_resids,
                                                index_validation=index_validation)
        self.volatility_model.build_model()
        self.volatility_model.fit()

    def fit_model(self):
        if self.return_model_type == 'AR' and self.volatility_model_type in ['arch', 'garch']:
            self.return_model = MixModel(stock_model=self,
                                         model_dict=self.model_dict)
            self.return_model.build_model()
            self.return_model.fit()

            self.evaluate_return_model()
            self.heteroscedasticity = self.return_model.fitted_conditional_volatility
            self.homoscedastic_return = self.stock.get_homoscedasticity_training_data()

            self.volatility_model = self.return_model

            self.homoskedasticity_test(step=1,
                                       mix_model=True)

        else:

            self.return_model = ReturnModel(stock_model=self,
                                            model_dict=self.model_dict['return_model'],
                                            return_train=self.data_training,
                                            index_validation=None)
            self.return_model.build_model()
            self.return_model.fit()

            self.volatility_model = VolatilityModel(stock_model=self,
                                                    model_dict=self.model_dict['volatility_model'],
                                                    resids=self.return_model.fitted_resids,
                                                    index_validation=None)
            self.volatility_model.build_model()
            self.volatility_model.fit()

            for step in range(1, self.model_dict['stock_model']['n_iteration'] + 1):
                self.fit_iteration(prev_return_model=self.return_model,
                                   prev_volatility_model=self.volatility_model,
                                   step=step)
                self.homoskedasticity_test(step=step)

    def homoskedasticity_test(self,
                              step=None,
                              mix_model=False):
        try :
            self.metrics_dict['init_skewness']
        except KeyError:
            training = self.stock.remove_normalisation(self.data_training).iloc[:self.stock.index_validation]
            self.metrics_dict['init_skewness'] = training.skew(axis=0)
            self.metrics_dict['init_kurtosis'] = training.kurtosis(axis=0)

        if step == self.model_dict['stock_model']['n_iteration']:
            if mix_model:
                heteroscedasticity_after = self.return_model.fitted_conditional_volatility
            else:
                heteroscedasticity_after = self.volatility_model.fitted_conditional_volatility
            homo_training_after = self.stock.remove_normalisation(self.stock.get_homoscedasticity_training_data(
                heteroscedasticity=heteroscedasticity_after)).iloc[:self.stock.index_validation]
            self.metrics_dict['final_skewness'] = homo_training_after.skew(axis=0)
            self.metrics_dict['final_kurtosis'] = homo_training_after.kurtosis(axis=0)
        else:
            lab = '_%i' % step
            homo_training = self.stock.remove_normalisation(self.homoscedastic_return).iloc[:self.stock.index_validation]
            self.metrics_dict['skewness'+lab] = homo_training.skew(axis=0)
            self.metrics_dict['kurtosis'+lab] = homo_training.kurtosis(axis=0)

    def evaluate(self,
                 return_model=None,
                 volatility_model=None,
                 evaluate_validation=True,
                 step=1):
        lab = '_%i' % (step)
        if return_model is None or volatility_model is None:
            return_model = self.return_model
            volatility_model = self.volatility_model
            lab = '_final'
        self.evaluate_return_model(model=return_model,
                                   evaluate_validation=evaluate_validation)

        r_pred_train = self.stock.remove_normalisation(return_model.get_return_pred_train())
        r_pred_validation = self.stock.remove_normalisation(return_model.get_return_pred_validation())
        r_true_train = self.stock.remove_normalisation(return_model.r_true_train)
        r_true_validation = self.stock.remove_normalisation(return_model.r_true_validation)

        self.metrics_dict['mse_train' + lab] = self.mse(r_pred=r_pred_train,
                                                        r_true=r_true_train)
        if evaluate_validation:
            self.metrics_dict['mse_validation' + lab] = self.mse(r_pred=r_pred_validation,
                                                                 r_true=r_true_validation)

        self.evaluate_volatility_model(model=volatility_model,
                                       evaluate_validation=evaluate_validation)

        vol_pred_train = self.stock.remove_normalisation(volatility_model.get_volatility_pred_train(),
                                                         type='volatility')
        vol_pred_validation = self.stock.remove_normalisation(volatility_model
                                                                       .get_volatility_pred_validation(),
                                                              type='volatility')

        self.metrics_dict['accuracy_train' + lab] = self.accuracy(r_pred=r_pred_train,
                                                                  r_true=r_true_train,
                                                                  vol_pred=vol_pred_train,
                                                                  quantile=None)
        if evaluate_validation:
            self.metrics_dict['accuracy_validation' + lab] = self.accuracy(r_pred=r_pred_validation,
                                                                           r_true=r_true_validation,
                                                                           vol_pred=vol_pred_validation,
                                                                           quantile=None)
        if evaluate_validation:
            self.metrics_dict['std_r_pred'] = r_pred_validation.std()
            self.metrics_dict['std_vol_pred'] = vol_pred_validation.std()

    def predict(self):
        self.return_model.predict(return_test=self.stock.test_return)
        r_pred_test = self.return_model.r_pred_test

        if self.return_model_type == 'AR' and self.volatility_model_type in ['arch', 'garch']:
            vol_pred_test = self.return_model.vol_pred_test
        else:
            resids_test = pd.Series(
                self.stock.test_return.iloc[len(self.stock.test_return) - len(r_pred_test):] - r_pred_test)
            self.volatility_model.predict(resids_test)
            vol_pred_test = self.volatility_model.vol_pred_test

        self.metrics_dict['mse_test'] = self.mse(r_pred=self.stock.remove_normalisation(r_pred_test),

                                                 r_true=self.stock.remove_normalisation(self.stock.test_return))
        self.metrics_dict['accuracy_test'] = self.accuracy(r_pred=self.stock.remove_normalisation(r_pred_test),
                                                           r_true=self.stock.remove_normalisation(
                                                               self.stock.test_return),
                                                           vol_pred=self.stock.remove_normalisation(vol_pred_test,
                                                                                                    type='volatility'),
                                                           quantile=None)

    @staticmethod
    def mse(r_pred,
            r_true):
        if r_pred is not None and r_true is not None:
            merged_df = pd.DataFrame(r_pred).merge(right=r_true, left_index=True, right_index=True, how='inner')
            columns = list(merged_df.columns)
            try:
                mse = mean_squared_error(y_true=merged_df[columns[0]],
                                         y_pred=merged_df[columns[1]])
            except ValueError:
                mse = None
            return mse
        return None

    @staticmethod
    def accuracy(r_pred,
                 r_true,
                 vol_pred,
                 quantile=1.96):
        if r_pred is not None and r_true is not None and vol_pred is not None:
            if quantile is None:
                r_pred_norm = (r_pred - r_pred.mean()) / r_pred.std()
                q_inf = r_pred_norm.quantile(0.05)
                q_sup = r_pred_norm.quantile(0.96)
            else:
                q_inf = quantile
                q_sup = quantile
            accuracy = []
            imin = max(min(r_pred.index), min(r_true.index), min(vol_pred.index))
            imax = min(max(r_pred.index), max(r_true.index), max(vol_pred.index))
            for i in range(imin, imax+1):
                mu = r_pred.loc[i]
                sigma = abs(vol_pred.loc[i])
                accuracy.append(int((r_true.loc[i] > mu + q_inf*sigma) & (r_true.loc[i] < mu + q_sup*sigma)))

            return np.mean(accuracy)
        return None

    def summary_fit(self):
        self.return_model.summary()
        self.volatility_model.summary()
        print(self.metrics_dict)

    def set_model_attribute(self,
                            model_type,
                            attribute,
                            value):
        self.model_dict[model_type][attribute] = value

    def reset_metrics(self):
        self.metrics_dict = dict()
        self.return_model.reset_model()
        self.volatility_model.reset_model()

    def evaluate_return_model(self,
                              model=None,
                              evaluate_validation=True):
        if model is None:
            model = self.return_model
        model.evaluate(evaluate_validation=evaluate_validation)

    def evaluate_volatility_model(self,
                                  model=None,
                                  evaluate_validation=True):
        if self.return_model_type == 'AR' and self.volatility_model_type in ['arch', 'garch']:
            pass
        else:
            if model is None:
                model = self.volatility_model
            model.evaluate(evaluate_validation=evaluate_validation)

    def save_model(self,
                   write_path=None):
        if write_path is None:
            write_path = 'models/%s_%i_%s_%i' % \
                         (self.model_dict['return_model']['type'],
                          self.model_dict['return_model']['ar_lags'],
                          self.model_dict['volatility_model']['type'],
                          self.model_dict['volatility_model']['ar_lags'])
        return_write_path = write_path + '_return.h5'
        volatility_write_path = write_path + '_vol.h5'

        self.return_model.save_model(return_write_path)
        if self.volatility_model is not None:
            self.volatility_model.save_model(volatility_write_path)

        f = open(write_path + '_metrics.txt', "w")
        f.write(str(self.metrics_dict))
        f.close()

    def read_model(self,
                   write_path=None):
        if write_path is None:
            write_path = 'models/%s_%i_%s_%i' % \
                         (self.model_dict['return_model']['type'],
                          self.model_dict['return_model']['ar_lags'],
                          self.model_dict['volatility_model']['type'],
                          self.model_dict['volatility_model']['ar_lags'])
        return_write_path = write_path + '_return.h5'
        volatility_write_path = write_path + '_vol.h5'

        self.return_model.fitted_model = load_model(return_write_path)
        self.volatility_model.fitted_model = load_model(volatility_write_path)
        self.reset_metrics()

        f = open(write_path + '_metrics.txt', "w")
        self.metrics_dict = eval(f.read())
        f.close()


class MixModel:
    def __init__(self,
                 stock_model,
                 model_dict):

        self.stock_model = stock_model
        self.return_model_type = model_dict['return_model']['type']
        self.volatility_model_type = model_dict['volatility_model']['type']
        self.index_validation = self.stock_model.stock.index_validation

        self.return_train = self.stock_model.data_training
        self.model = None
        self.model_dict = model_dict
        self.ar_lags = model_dict['return_model']['ar_lags']
        self.overwrite_params = self.model_dict['return_model']['params']

        self.fitted_model = None
        self.fitted_resids = None
        self.fitted_conditional_volatility = None

        self.r_pred_train = None
        self.r_pred_validation = None
        self.vol_pred_train = None
        self.vol_pred_validation = None

        self.r_true_train = self.return_train[self.ar_lags:self.index_validation]
        self.r_true_validation = self.return_train[self.index_validation + self.ar_lags + 1:]

        self.r_pred_test = None
        self.vol_pred_test = None

    def build_model(self):
        model_dict = {'mean': self.return_model_type,
                      'lags': self.ar_lags,
                      'vol': self.volatility_model_type,
                      'dist': 'ged',
                      'constant': self.model_dict['return_model']['constant'],
                      'rescale': False}
        self.model = arch.univariate.ARX(y=self.return_train,
                                         rescale=model_dict['rescale'],
                                         constant=model_dict['constant'],
                                         lags=model_dict['lags'])
        if self.volatility_model_type == 'arch':
            model_dict['p'] = self.model_dict['volatility_model']['ar_lags']
            self.model.volatility = arch.univariate.ARCH(p=model_dict['p'])
        elif self.volatility_model_type == 'garch':
            model_dict['p'] = self.model_dict['volatility_model']['ar_lags']
            model_dict['q'] = self.model_dict['volatility_model']['ma_lags']
            self.model.volatility = arch.univariate.GARCH(p=model_dict['p'], q=model_dict['q'])

        if model_dict['dist'] == 'ged':
            self.model.distribution = arch.univariate.GeneralizedError()
        elif model_dict['dist'] == 'normal':
            self.model.distribution = arch.univariate.Normal()
        # self.model = arch.arch_model(y=self.return_train,
        #                              rescale=False,
        #                              **model_dict)

    def fit(self):
        self.fitted_model = self.model.fit(update_freq=5,
                                           last_obs=self.stock_model.stock.index_validation)

    def summary(self):
        print(self.fitted_model.summary())

    def get_return_pred_train(self):
        return self.r_pred_train

    def get_return_pred_validation(self):
        return self.r_pred_validation

    def get_volatility_pred_train(self):
        return self.vol_pred_train

    def get_volatility_pred_validation(self):
        return self.vol_pred_validation

    def evaluate(self,
                 evaluate_validation=True):
        if self.overwrite_params is None:
            params = self.fitted_model.params
        else:
            params = self.overwrite_params
        forecast = self.fitted_model.forecast(horizon=1,
                                              params=params,
                                              start=self.ar_lags)

        forecast_return = forecast.mean.dropna()
        forecast_volatility = forecast.variance.dropna()

        self.r_pred_train = forecast_return.iloc[:self.index_validation]
        self.vol_pred_train = forecast_volatility.iloc[:self.index_validation]

        if evaluate_validation:
            self.r_pred_validation = forecast_return.iloc[self.index_validation + 1:]
            self.vol_pred_validation = forecast_volatility.iloc[self.index_validation + 1:]

            self.fitted_conditional_volatility = pd.concat([self.vol_pred_train, self.vol_pred_validation], axis=0)
        else:
            self.fitted_conditional_volatility = self.vol_pred_train

    def predict(self,
                return_test):
        model_dict = {'mean': self.return_model_type,
                      'lags': self.ar_lags,
                      'vol': self.volatility_model_type,
                      'dist': 'ged',
                      'constant': self.model_dict['return_model']['constant'],
                      'rescale': False}
        model = arch.univariate.ARX(y=return_test,
                                    rescale=model_dict['rescale'],
                                    constant=model_dict['constant'],
                                    lags=model_dict['lags'])
        if self.volatility_model_type == 'arch':
            model_dict['p'] = self.model_dict['volatility_model']['ar_lags']
            model.volatility = arch.univariate.ARCH(p=model_dict['p'])
        elif self.volatility_model_type == 'garch':
            model_dict['p'] = self.model_dict['volatility_model']['ar_lags']
            model_dict['q'] = self.model_dict['volatility_model']['ma_lags']
            model.volatility = arch.univariate.GARCH(p=model_dict['p'], q=model_dict['q'])

        if model_dict['dist'] == 'ged':
            model.distribution = arch.univariate.GeneralizedError()
        elif model_dict['dist'] == 'normal':
            model.distribution = arch.univariate.Normal()

        if self.overwrite_params is None:
            params = self.fitted_model.params
        else:
            params = self.overwrite_params
        model = model.fit()
        forecast = model.forecast(horizon=1,
                                  params=params,
                                  start=self.ar_lags)

        self.r_pred_test = forecast.mean.dropna()
        self.vol_pred_test = forecast.variance.dropna()

    def set_return_train(self,
                         S):
        self.return_train = S
        self.r_true_train = self.return_train[self.ar_lags:self.index_validation]
        self.r_true_validation = self.return_train[self.index_validation + self.ar_lags + 1:]
        self.overwrite_params = self.fitted_model.params
        self.build_model()
        self.fit()

    def set_params(self,
                   params):
        self.overwrite_params = params

    def reset_model(self):
        self.r_pred_train = None
        self.r_pred_validation = None
        self.vol_pred_train = None
        self.vol_pred_validation = None

    def save_model(self,
                   write_path):
        pass


class ReturnModel:
    def __init__(self,
                 stock_model,
                 model_dict,
                 return_train,
                 index_validation=None,
                 endogenous_feature=None):

        self.stock_model = stock_model
        self.model_dict = model_dict
        self.model_type = self.model_dict['type']
        self.return_train = return_train
        if index_validation is None:
            self.index_validation = len(self.return_train) - 1
        else:
            self.index_validation = index_validation - self.return_train.index[0]

        self.generator_train = None
        self.generator_validation = None
        self.model = None
        self.ar_lags = self.model_dict['ar_lags']
        self.ma_lags = self.model_dict['ma_lags']

        self.fitted_model = None
        self.fitted_resids = None
        self.all_resids = None

        self.overwrite_params = self.model_dict['params']
        self.endogenous_feature = endogenous_feature

        self.r_pred_train = None
        self.r_pred_validation = None
        self.r_true_train = self.return_train[self.ar_lags:self.index_validation]
        self.r_true_validation = self.return_train[self.index_validation + self.ar_lags + 1:]

        self.r_pred_test = None

    def build_model(self):
        if self.model_type == 'AR':
            self.model = arch.univariate.ARX(y=self.return_train.ravel(),
                                             lags=self.ar_lags,
                                             constant=self.model_dict['constant'],
                                             rescale=False)
            # else:
            #     self.model = arch.univariate.ARX(y=self.return_train.ravel(),
            #                                      x=self.endogenous_feature,
            #                                      lags=self.ar_lags,
            #                                      constant=self.model_dict['constant'],
            #                                      rescale=False)

        elif self.model_type == 'NN':
            self.model = Sequential()
            self.model.add(Dense(units=self.model_dict['layers'][0],
                                 activation=self.model_dict['activation'],
                                 input_dim=self.ar_lags))
            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.model_dict['layers'][1],
                                 activation='linear'))
            self.model.compile(loss="mse",
                               optimizer="adam")

        elif self.model_type == 'RNN':
            self.model = Sequential()
            self.model.add(SimpleRNN(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                use_bias=self.model_dict['constant'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                output_dim=self.model_dict['layers'][1]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

        elif self.model_type == 'GRU':
            self.model = Sequential()

            self.model.add(GRU(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                units=self.model_dict['layers'][1]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

        elif self.model_type == 'DoubleGRU':
            self.model = Sequential()

            self.model.add(GRU(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                return_sequences=True))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(GRU(
                units=self.model_dict['layers'][1],
                activation=self.model_dict['activation'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                units=self.model_dict['layers'][2]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

        elif self.model_type == 'LSTM':
            self.model = Sequential()

            self.model.add(LSTM(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                units=self.model_dict['layers'][1]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

        elif self.model_type == 'DoubleLSTM':
            self.model = Sequential()

            self.model.add(LSTM(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                return_sequences=True))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(LSTM(
                units=self.model_dict['layers'][1],
                activation=self.model_dict['activation'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                units=self.model_dict['layers'][2]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

    def fit(self):
        if self.model_type == 'AR':
            self.fitted_model = self.model.fit(update_freq=5,
                                               last_obs=self.index_validation)
            params_index = self.ar_lags + int(self.model_dict['constant'])
            self.fitted_resids = pd.Series(self.model.resids(
                params=self.fitted_model.params[:params_index]))

        elif self.model_type in ['NN', 'RNN', 'GRU', 'LSTM', 'DoubleGRU', 'DoubleLSTM']:
            # TODO: add endogenous variable with timeseries
            if self.model_type in ['RNN', 'GRU', 'LSTM', 'DoubleGRU', 'DoubleLSTM']:
                data = self.return_train.values.reshape((len(self.return_train), 1))
                self.generator_train = TimeseriesGenerator(data=data,
                                                           targets=data[:, 0],
                                                           length=self.ar_lags,
                                                           batch_size=8,
                                                           end_index=self.index_validation)
                self.generator_validation = TimeseriesGenerator(data=data,
                                                                targets=data[:, 0],
                                                                length=self.ar_lags,
                                                                start_index=self.index_validation)
            else:
                self.generator_train = TimeseriesGenerator(data=self.return_train.ravel(),
                                                           targets=self.return_train.ravel(),
                                                           length=self.ar_lags,
                                                           batch_size=8,
                                                           end_index=self.index_validation)
                self.generator_validation = TimeseriesGenerator(data=self.return_train.ravel(),
                                                                targets=self.return_train.ravel(),
                                                                length=self.ar_lags,
                                                                start_index=self.index_validation)

            self.model.fit_generator(self.generator_train,
                                     steps_per_epoch=1,
                                     epochs=200,
                                     verbose=0,
                                     use_multiprocessing=True)
            self.fitted_model = self.model

            y_pred_train = self.fitted_model.predict_generator(self.generator_train, verbose=0)

            self.fitted_resids = y_pred_train[:, 0] - self.return_train.iloc[self.ar_lags:self.index_validation]

    def evaluate(self,
                 evaluate_validation=True):
        if self.all_resids is None:
            if self.model_type == 'AR':
                if self.overwrite_params is None:
                    params = self.fitted_model.params
                else:
                    params = self.overwrite_params
                forecast = self.fitted_model.forecast(horizon=1,
                                                      params=params,
                                                      start=self.ar_lags) \
                    .mean.dropna()
                self.r_pred_train = forecast.iloc[:self.index_validation]
                if evaluate_validation:
                    self.r_pred_validation = forecast.iloc[self.index_validation+1:]

            elif self.model_type in ['NN', 'RNN', 'GRU', 'LSTM', 'DoubleGRU', 'DoubleLSTM']:
                self.r_pred_train = pd.Series(self.fitted_model.predict_generator(self.generator_train)[:, 0],
                                              index=self.return_train.index[self.ar_lags:self.index_validation])

                if evaluate_validation:
                    self.r_pred_validation = pd.Series(self.fitted_model.predict_generator(self.generator_validation)[:, 0],
                                                       index=self.return_train.index[
                                                             self.index_validation + self.ar_lags + 1:])
            if evaluate_validation:
                self.all_resids = pd.concat([self.r_pred_train, self.r_pred_validation], axis=0)
            else:
                self.all_resids = self.r_pred_train

    def predict(self,
                return_test):
        if self.model_type == 'AR':
            model = arch.univariate.ARX(y=return_test,
                                        lags=self.ar_lags,
                                        constant=self.model_dict['constant'],
                                        rescale=False)
            if self.overwrite_params is None:
                params = self.fitted_model.params
            else:
                params = self.overwrite_params
            model = model.fit()
            self.r_pred_test = model.forecast(params=params,
                                              horizon=1,
                                              start=self.ar_lags).mean.dropna()
            self.r_pred_test = self.r_pred_test[list(self.r_pred_test.columns)[0]]

        elif self.model_type == 'NN':
            generator_test = TimeseriesGenerator(data=return_test.ravel(),
                                                 targets=return_test.ravel(),
                                                 length=self.ar_lags)
            self.r_pred_test = pd.Series(np.exp(self.fitted_model.predict_generator(generator_test)[:, 0]),
                                         index=return_test.index[self.ar_lags + 1:])

        elif self.model_type in ['RNN', 'GRU', 'LSTM', 'DoubleGRU', 'DoubleLSTM']:
            data = return_test.values.reshape((len(return_test), 1))

            generator_test = TimeseriesGenerator(data=data,
                                                 targets=data[:, 0],
                                                 length=self.ar_lags)

            self.r_pred_test = pd.Series(np.exp(self.fitted_model.predict_generator(generator_test)[:, 0]),
                                         index=return_test.index[self.ar_lags + 1:])

    def get_return_pred_train(self):
        return self.r_pred_train

    def get_return_pred_validation(self):
        return self.r_pred_validation

    def summary(self):
        print(self.fitted_model.summary())

    def reset_model(self):
        self.all_resids = None

    def set_return_train(self,
                         S):
        self.return_train = S
        self.r_true_train = self.return_train[self.ar_lags:self.index_validation]
        self.r_true_validation = self.return_train[self.index_validation + self.ar_lags + 1:]
        self.all_resids = None
        if self.model_type == 'AR':
            self.overwrite_params = self.fitted_model.params
            self.build_model()
            self.fit()
        elif self.model_type == 'NN':
            self.generator_train = TimeseriesGenerator(data=self.return_train.ravel(),
                                                       targets=self.return_train.ravel(),
                                                       length=self.ar_lags,
                                                       batch_size=8,
                                                       end_index=self.index_validation)
            self.generator_validation = TimeseriesGenerator(data=self.return_train.ravel(),
                                                            targets=self.return_train.ravel(),
                                                            length=self.ar_lags,
                                                            start_index=self.index_validation)
        elif self.model_type in ['RNN', 'GRU', 'LSTM', 'DoubleGRU', 'DoubleLSTM']:
            data = self.return_train.values.reshape((len(self.return_train), 1))
            self.generator_train = TimeseriesGenerator(data=data,
                                                       targets=data[:, 0],
                                                       length=self.ar_lags,
                                                       batch_size=8,
                                                       end_index=self.index_validation)
            self.generator_validation = TimeseriesGenerator(data=data,
                                                            targets=data[:, 0],
                                                            length=self.ar_lags,
                                                            start_index=self.index_validation)

    def save_model(self,
                   write_path):
        if self.fitted_model is not None and self.model_type != 'AR':
            self.fitted_model.save(write_path)


class VolatilityModel:
    def __init__(self,
                 stock_model,
                 model_dict,
                 resids,
                 index_validation=None):

        self.stock_model = stock_model
        self.model_dict = model_dict
        self.model_type = self.model_dict['type']
        self.resids = resids

        self.ar_lags = self.model_dict['ar_lags']
        self.ma_lags = self.model_dict['ma_lags']

        if index_validation is None:
            self.index_validation = len(self.resids) - 1
        else:
            self.index_validation = index_validation - self.resids.index[0]

        self.generator_train = None
        self.generator_validation = None
        self.model = None
        self.fitted_model = None

        self.fitted_conditional_volatility = None

        self.vol_pred_train = None
        self.vol_pred_validation = None
        self.vol_true_train = None
        self.vol_true_validation = None

        self.vol_pred_test = None

    def build_model(self):
        if self.model_type == 'constant':
            pass
        elif self.model_type in ['arch', 'garch']:
            model_dict = {'mean': 'zero',
                          'vol': self.model_type,
                          'dist': 'ged'}
            if self.model_type == 'arch':
                model_dict['p'] = self.ar_lags
            elif self.model_type == 'garch':
                model_dict['p'] = self.ar_lags
                model_dict['q'] = self.ma_lags
            self.model = arch.arch_model(y=self.resids.ravel(),
                                         rescale=False,
                                         **model_dict)
        elif self.model_type == 'NN':
            self.model = Sequential()
            self.model.add(Dense(units=self.model_dict['layers'][0],
                                 activation=self.model_dict['activation'],
                                 input_dim=self.ar_lags))
            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.model_dict['layers'][1],
                                 activation='linear'))
            self.model.compile(loss="mse",
                               optimizer="adam")

        elif self.model_type == 'RNN':
            self.model = Sequential()
            self.model.add(SimpleRNN(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                use_bias=self.model_dict['constant'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                units=self.model_dict['layers'][1]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

        elif self.model_type == 'GRU':
            self.model = Sequential()

            self.model.add(GRU(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                units=self.model_dict['layers'][1]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

        elif self.model_type == 'DoubleGRU':
            self.model = Sequential()

            self.model.add(GRU(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                return_sequences=True))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(GRU(
                units=self.model_dict['layers'][1],
                activation=self.model_dict['activation'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                units=self.model_dict['layers'][2]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

        elif self.model_type == 'LSTM':
            self.model = Sequential()

            self.model.add(LSTM(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                units=self.model_dict['layers'][1]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

        elif self.model_type == 'DoubleLSTM':
            self.model = Sequential()

            self.model.add(LSTM(
                units=self.model_dict['layers'][0],
                input_shape=(self.ar_lags, 1),
                activation=self.model_dict['activation'],
                return_sequences=True))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(LSTM(
                units=self.model_dict['layers'][1],
                activation=self.model_dict['activation'],
                return_sequences=False))

            if self.model_dict['dropout']:
                self.model.add(Dropout(0.2))

            self.model.add(Dense(
                units=self.model_dict['layers'][2]))
            self.model.add(Activation("linear"))

            self.model.compile(loss="mse",
                               optimizer="rmsprop")

    def fit(self):
        self.vol_true_train = self.resids.iloc[self.ar_lags:self.index_validation]
        self.vol_true_validation = self.resids.iloc[self.index_validation + self.ar_lags + 1:]

        if self.model_type == 'constant':
            self.fitted_conditional_volatility = pd.Series(self.resids.std(),
                                                           index=self.resids.index)

        elif self.model_type in ['arch', 'garch']:
            self.fitted_model = self.model.fit(update_freq=5)

            self.fitted_conditional_volatility = self.fitted_model.conditional_volatility.dropna().apply(
                lambda var: var ** 0.5)

        elif self.model_type in ['NN', 'RNN', 'GRU', 'LSTM', 'DoubleGRU', 'DoubleLSTM']:
            log_resids_sq = np.log(self.resids.apply(lambda resid: resid ** 2).values)
            if len(log_resids_sq.shape) > 1:
                log_resids_sq = log_resids_sq[:, 0]
            if self.model_type in ['RNN', 'GRU', 'LSTM', 'DoubleGRU', 'DoubleLSTM']:
                data = log_resids_sq.reshape((len(log_resids_sq), 1))
                self.generator_train = TimeseriesGenerator(data=data,
                                                           targets=data[:, 0],
                                                           length=self.ar_lags,
                                                           batch_size=8,
                                                           end_index=self.index_validation)
                self.generator_validation = TimeseriesGenerator(data=data,
                                                                targets=data[:, 0],
                                                                length=self.ar_lags,
                                                                start_index=self.index_validation)
            else:
                self.generator_train = TimeseriesGenerator(data=log_resids_sq,
                                                           targets=log_resids_sq,
                                                           length=self.ar_lags,
                                                           batch_size=8,
                                                           end_index=self.index_validation)
                self.generator_validation = TimeseriesGenerator(data=log_resids_sq,
                                                                targets=log_resids_sq,
                                                                length=self.ar_lags,
                                                                start_index=self.index_validation)
            self.model.fit_generator(self.generator_train,
                                     steps_per_epoch=1,
                                     epochs=200,
                                     verbose=0,
                                     use_multiprocessing=True)
            self.fitted_model = self.model

            y_pred = np.exp(self.fitted_model.predict_generator(self.generator_train, verbose=0))
            self.fitted_conditional_volatility = pd.Series(y_pred[:, 0]).apply(lambda var: abs(var) ** 0.5)

    def evaluate(self,
                 evaluate_validation=True):
        if self.vol_pred_train is None:
            if self.model_type == 'constant':
                self.vol_pred_train = pd.Series(self.resids.std(),
                                                index=pd.Index(range(self.ar_lags, self.index_validation)))
                if evaluate_validation:
                    self.vol_pred_validation = pd.Series(self.resids.std(),
                                                         index=pd.Index(range(self.index_validation + self.ar_lags + 1,
                                                                              len(self.resids))))
            if self.model_type in ['arch', 'garch']:
                self.vol_pred_train = self.fitted_model.forecast(end=self.index_validation,
                                                                 horizon=1) \
                    .volatility.dropna()
                if evaluate_validation:
                    self.vol_pred_validation = self.fitted_model.forecast(start=self.index_validation + 1,
                                                                          horizon=1) \
                        .volatility.dropna()

            elif self.model_type in ['NN', 'RNN', 'GRU', 'LSTM', 'DoubleGRU', 'DoubleLSTM']:
                self.vol_pred_train = pd.Series(np.exp(self.fitted_model.predict_generator(self.generator_train)[:, 0]),
                                                index=self.resids.index[self.ar_lags:self.index_validation])

                if evaluate_validation:
                    self.vol_pred_validation = pd.Series(
                        np.exp(self.fitted_model.predict_generator(self.generator_validation)[:, 0]),
                        index=self.resids.index[
                              self.index_validation + self.ar_lags + 1:])

    def predict(self,
                resids_test):
        if self.model_type in ['arch', 'garch']:
            model_dict = {'mean': 'zero',
                          'vol': self.model_type,
                          'dist': 'ged'}
            if self.model_type == 'arch':
                model_dict['p'] = self.ar_lags
            elif self.model_type == 'garch':
                model_dict['p'] = self.ar_lags
                model_dict['q'] = self.ma_lags
            model = arch.arch_model(y=resids_test.ravel(),
                                    rescale=False,
                                    **model_dict)
            model = model.fit()
            self.vol_pred_test = model.forecast(params=self.fitted_model.params).variance.dropna()
            self.vol_pred_test = self.vol_pred_test[list(self.vol_pred_test.columns)[0]]

        elif self.model_type == 'constant':
            self.vol_pred_test = pd.Series(self.resids.std(),
                                           index=resids_test.index)

        else:
            log_resids_sq = np.log(resids_test.apply(lambda resid: resid ** 2).values)
            if len(log_resids_sq.shape) > 1:
                log_resids_sq = log_resids_sq[:, 0]

            if self.model_type == 'NN':
                generator_test = TimeseriesGenerator(data=log_resids_sq,
                                                     targets=log_resids_sq,
                                                     length=self.ar_lags)

            elif self.model_type in ['RNN', 'GRU', 'LSTM', 'DoubleGRU', 'DoubleLSTM']:
                data = log_resids_sq.reshape((len(log_resids_sq), 1))

                generator_test = TimeseriesGenerator(data=data,
                                                     targets=data[:, 0],
                                                     length=self.ar_lags)

            pred = np.exp(self.fitted_model.predict_generator(generator_test)[:, 0])
            self.vol_pred_test = pd.Series(pred,
                                           index=resids_test.index[len(resids_test) - len(pred):])

    def get_volatility_pred_train(self):
        return self.vol_pred_train

    def get_volatility_pred_validation(self):
        return self.vol_pred_validation

    def reset_model(self):
        self.vol_pred_train = None

    def summary(self):
        print(self.fitted_model.summary())

    def save_model(self,
                   write_path):
        if self.fitted_model is not None and self.model_type not in ['constant', 'arch', 'garch']:
            self.fitted_model.save(write_path)


def find_opt_params(model_dict,
                    model_type_to_change,
                    attribute_to_change,
                    value_list,
                    n_iter=5,
                    write_path='grid_search_params.xlsx'):
    n_steps = len(value_list)*n_iter
    step = 0
    test_dict = dict()
    for value in value_list:
        model_test_dict = dict()
        model_dict[model_type_to_change][attribute_to_change] = value
        model_write_path = 'models_p/%s_%i_%s_%i' % \
                           (model_dict['return_model']['type'],
                            model_dict['return_model']['ar_lags'],
                            model_dict['volatility_model']['type'],
                            model_dict['volatility_model']['ar_lags'])
        for i in range(n_iter):
            step += 1
            start = time.time()

            stock = Stock(stock_name='Bnp Paribas',
                          data_path='clean_data.csv',
                          return_column='log_return',
                          model_dict=model_dict)

            stock.fit_model()
            stock.model.evaluate()
            stock.model.save_model(write_path=model_write_path+'_%i' % i)

            model_test_dict[i] = stock.get_metrics()
            stock.reset_metrics()

            end = time.time()
            print("Step %i / %i, Time remaining (min) : %.2f" % (step, n_steps, (end-start)*(n_steps-step)/60))
        k_opt, i_opt = min([[model_test_dict[i]['final_kurtosis'], i] for i in range(5)])
        mean_k = np.mean([model_test_dict[i]['final_kurtosis'] for i in range(5)])
        std_k = np.std([model_test_dict[i]['final_kurtosis'] for i in range(5)]) / 5
        test_dict[value] = model_test_dict[i_opt]
        test_dict[value]['mean_kurtosis'] = mean_k
        test_dict[value]['std_kurtosis'] = std_k
        test_dict[value]['num_model'] = i_opt

    metrics_list = list(test_dict[value_list[0]].keys())
    df_dict = {metric: [test_dict[key][metric] for key in value_list] for metric in metrics_list}
    pd.DataFrame.from_dict(df_dict).to_excel(write_path)


def test_models(model_dict,
                model_dict_change,
                write_path='best_models.xlsx'):

    key1_list = list(model_dict_change.keys())
    key2_list = list(model_dict_change[key1_list[0]])
    n_steps = len(model_dict_change[key1_list[0]][key2_list[0]])
    test_dict = dict()
    for step in range(n_steps):
        for key1 in key1_list:
            for key2 in model_dict_change[key1].keys():
                model_dict[key1][key2] = model_dict_change[key1][key2][step]

        print('Step : %i / %i' % (step, n_steps))
        model_write_path = 'models_comparision/%s_%i_%s_%i' % \
                           (model_dict['return_model']['type'],
                            model_dict['return_model']['ar_lags'],
                            model_dict['volatility_model']['type'],
                            model_dict['volatility_model']['ar_lags'])

        stock = Stock(stock_name='Bnp Paribas',
                      data_path='clean_data.csv',
                      return_column='log_return',
                      model_dict=model_dict)

        stock.fit_model()
        stock.model.evaluate()
        stock.model.predict()
        stock.model.save_model(write_path=model_write_path)
        test_dict[step] = stock.get_metrics()
        test_dict[step]['return_model'] = '%s_%i' % (model_dict['return_model']['type'],
                                                     model_dict['return_model']['ar_lags'])
        test_dict[step]['volatility_model'] = '%s_%i' % (model_dict['volatility_model']['type'],
                                                         model_dict['volatility_model']['ar_lags'])

    metrics_list = list(test_dict[0].keys())
    df_dict = {metric: [test_dict[key][metric] for key in range(n_steps)] for metric in metrics_list}
    pd.DataFrame.from_dict(df_dict).to_excel(write_path)


if __name__ == '__main__':

    pd.set_option('mode.chained_assignment', None)

    seed(1)
    set_random_seed(1)
    data_path = 'clean_data.csv'
    return_column = 'log_return'

    # params_eviews = pd.Series(data=[1.8e-18, 1.59e-18, 4.16e-8, 0.516, 0.166, 0.133, 1.01],
    #                           index=['Const', 'log_return[-1]', 'omega', 'alpha[1]', 'alpha[2]', 'alpha[3]', 'nu'])

    params_eviews = pd.Series(data=[0.002445, 13.7412, 5.19641, 5.887933, 6.158594, 1.01],
                              index=['log_return[-1]', 'omega', 'alpha[1]', 'alpha[2]', 'alpha[3]', 'nu'])

    # model_type_list = ['AR', 'arch', 'garch', 'NN', 'RNN', 'GRU', 'DoubleGRU', 'LSTM', 'DoubleLSTM']

    model_dict = {
        'return_model': {
            'type': 'AR',
            'constant': True,
            'params': params_eviews,
            'ar_lags': 1,
            'ma_lags': 5,
            'layers': [10, 1],
            'activation': 'sigmoid',
            'dropout': True
        },
        'volatility_model': {
            'type': 'arch',
            'constant': True,
            'params': None,
            'ar_lags': 3,
            'ma_lags': 5,
            'layers': [10, 1],
            'activation': 'sigmoid',
            'dropout': True
        },
        'stock_model': {
            'n_iteration': 1
        }
    }
    #
    # model_dict_change = {
    #     'return_model': {
    #         'type': ['RNN', 'NN', 'NN', 'LSTM'],
    #         'ar_lags': [6, 1, 10, 6],
    #     },
    #     'volatility_model': {
    #         'type': ['LSTM', 'NN', 'LSTM', 'LSTM'],
    #         'ar_lags': [6, 2, 10, 6]
    #     }
    # }
    #
    # test_models(model_dict=model_dict,
    #             model_dict_change=model_dict_change,
    #             write_path='best_models_vol2.xlsx')

    # model_dict_change = {
    #     'return_model': {
    #         'type': ['AR', 'NN', 'GRU', 'LSTM'],
    #         'ar_lags': [1, 6, 6, 10],
    #     },
    #     'volatility_model': {
    #         'type': ['constant', 'constant', 'constant', 'constant'],
    #         'ar_lags': [3, 1, 2, 6]
    #     }
    # }
    #
    # test_models(model_dict=model_dict,
    #             model_dict_change=model_dict_change,
    #             write_path='best_models_constant.xlsx')
    #
    # model_dict_change = {
    #     'return_model': {
    #         'type': ['DoubleGRU', 'DoubleLSTM', 'DoubleGRU', 'DoubleGRU', 'DoubleLSTM', 'DoubleLSTM', 'DoubleGRU',
    #                  'DoubleLSTM'],
    #         'ar_lags': [6, 10, 6, 10, 10, 6, 10, 10],
    #         'layers': [[10, 20, 1]] * 6 + [[50, 100, 1]] * 2
    #     },
    #     'volatility_model': {
    #         'type': ['constant', 'constant', 'DoubleGRU', 'DoubleGRU', 'DoubleLSTM', 'DoubleLSTM', 'DoubleGRU',
    #                  'DoubleLSTM'],
    #         'ar_lags': [1, 1, 6, 6, 10, 6, 10, 10],
    #         'layers': [[10, 20, 1]] * 6 + [[50, 100, 1]] * 2
    #     }
    # }
    #
    # test_models(model_dict=model_dict,
    #             model_dict_change=model_dict_change,
    #             write_path='best_models_double.xlsx')

    stock = Stock(stock_name='Bnp Paribas',
                  data_path=data_path,
                  return_column=return_column,
                  model_dict=model_dict)

    stock.fit_model()
    stock.model.evaluate()
    stock.model.predict()
    stock.model.summary_fit()

    #     stock.model.save_model(write_path=model_write_path)

    #
    # find_opt_params(model_dict=model_dict,
    #                 model_type_to_change='return_model',
    #                 attribute_to_change='ar_lags',
    #                 value_list=range(1, 11),
    #                 write_path='grid_search_gru_p_gru_5.xlsx')

    import os
    files = os.listdir('models_comparision')
    files = [file for file in files if 'metrics' in file]
    test_dict = dict()
    for i, file in enumerate(files):
        path = 'models_comparision/'+file
        f = open(path, 'r')
        test_dict[i] = eval(f.read())
        f.close()
        test_dict[i]['model'] = file[:-12]


