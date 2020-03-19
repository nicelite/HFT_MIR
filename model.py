import pandas as pd
import arch
import numpy as np
from math import log, exp, sqrt

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
                             series):
        return series * self.std_train + self.mean_train

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
        hetero_values = self.model.heteroscedasticity.values
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

    def fit_model(self):
        if self.return_model_type == 'AR' and self.volatility_model_type in ['arch', 'garch']:
            self.return_model = MixModel(stock_model=self,
                                         model_dict=self.model_dict)
            self.return_model.build_model()
            self.return_model.fit()

            self.evaluate_return_model()
            self.heteroscedasticity = self.return_model.fitted_conditional_variance
            self.homoscedastic_return = self.stock.get_homoscedasticity_training_data()

            self.volatility_model = self.return_model

        else:
            return_model = ReturnModel(stock_model=self,
                                       model_dict=self.model_dict['return_model'],
                                       return_train=self.data_training,
                                       index_validation=None)
            return_model.build_model()
            return_model.fit()

            volatility_model = VolatilityModel(stock_model=self,
                                               model_dict=self.model_dict['volatility_model'],
                                               resids=return_model.fitted_resids,
                                               index_validation=None)
            volatility_model.build_model()
            volatility_model.fit()

            self.heteroscedasticity = volatility_model.fitted_conditional_volatility
            self.homoscedastic_return = self.stock.get_homoscedasticity_training_data()

            self.return_model = ReturnModel(stock_model=self,
                                            model_dict=self.model_dict['return_model'],
                                            return_train=self.homoscedastic_return,
                                            index_validation=self.stock.index_validation)
            self.return_model.build_model()
            self.return_model.fit()

            self.return_model.set_return_train(S=self.data_training)
            self.evaluate_return_model()

            self.volatility_model = VolatilityModel(stock_model=self,
                                                    model_dict=self.model_dict['volatility_model'],
                                                    resids=self.return_model.all_resids,
                                                    index_validation=self.stock.index_validation)
            self.volatility_model.build_model()
            self.volatility_model.fit()

    def homoskedasticity_test(self):
        # bp_test = het_breuschpagan(resid=self.return_model.fitted_resids,
        #                            exog_het=self.return_model.r_true_train)
        # print(bp_test)
        training = self.stock.remove_normalisation(self.data_training)
        self.metrics_dict['init_skewness'] = training.skew(axis=0)
        self.metrics_dict['init_kurtosis'] = training.kurtosis(axis=0)

        homo_training = self.stock.remove_normalisation(self.homoscedastic_return)
        self.metrics_dict['final_skewness'] = homo_training.skew(axis=0)
        self.metrics_dict['final_kurtosis'] = homo_training.kurtosis(axis=0)

    def evaluate(self):
        self.evaluate_return_model()
        self.metrics_dict['mse_train'] = self.mse(r_pred=self.return_model.get_return_pred_train(),
                                                  r_true=self.return_model.r_true_train)
        self.metrics_dict['mse_validation'] = self.mse(r_pred=self.return_model.get_return_pred_validation(),
                                                       r_true=self.return_model.r_true_validation)

        self.evaluate_volatility_model()
        self.metrics_dict['accuracy_train'] = self.accuracy(r_pred=self.return_model.get_return_pred_train(),
                                                            r_true=self.return_model.r_true_train,
                                                            vol_pred=self.volatility_model.get_volatility_pred_train())

        self.metrics_dict['accuracy_validation'] = self.accuracy(r_pred=self.return_model.get_return_pred_validation(),
                                                                 r_true=self.return_model.r_true_validation,
                                                                 vol_pred=self.volatility_model
                                                                 .get_volatility_pred_validation())

    @staticmethod
    def mse(r_pred,
            r_true):
        merged_df = pd.DataFrame(r_pred).merge(right=r_true, left_index=True, right_index=True, how='inner')
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
            sigma = sqrt(abs(vol_pred.loc[i]))
            accuracy.append(int((r_true.loc[i] > mu-quantile*sigma) & (r_true.loc[i] < mu+quantile*sigma)))

        return np.mean(accuracy)

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

    def evaluate_return_model(self,
                              model=None):
        if model is None:
            model = self.return_model
        model.evaluate()

    def evaluate_volatility_model(self,
                                  model=None):
        if self.return_model_type == 'AR' and self.volatility_model_type in ['arch', 'garch']:
            pass
        else:
            if model is None:
                model = self.volatility_model
            model.evaluate()


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
        self.overwrite_params = None

        self.fitted_model = None
        self.fitted_resids = None
        self.fitted_conditional_variance = None

        self.r_pred_train = None
        self.r_pred_validation = None
        self.vol_pred_train = None
        self.vol_pred_validation = None

        self.r_true_train = self.return_train[self.ar_lags:self.index_validation]
        self.r_true_validation = self.return_train[self.index_validation + self.ar_lags + 1:]

    def build_model(self):
        model_dict = {'mean': self.return_model_type,
                      'lags': self.ar_lags,
                      'vol': self.volatility_model_type,
                      'dist': 'ged'}
        if self.volatility_model_type == 'arch':
            model_dict['p'] = self.model_dict['volatility_model']['ar_lags']
        elif self.volatility_model_type == 'garch':
            model_dict['p'] = self.model_dict['volatility_model']['ar_lags']
            model_dict['q'] = self.model_dict['volatility_model']['ma_lags']
        self.model = arch.arch_model(y=self.return_train,
                                     rescale=False,
                                     **model_dict)

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

    def evaluate(self):
        if self.overwrite_params is None:
            params = self.fitted_model.params
        else:
            params = self.overwrite_params
        forecast = self.fitted_model.forecast(horizon=1,
                                              params=params,
                                              start=self.ar_lags)

        forecast_return = forecast.mean.dropna()
        self.r_pred_train = forecast_return.iloc[:self.index_validation]
        self.r_pred_validation = forecast_return.iloc[self.index_validation + 1:]

        forecast_volatility = forecast.variance.dropna()
        self.vol_pred_train = forecast_volatility.iloc[:self.index_validation]
        self.vol_pred_validation = forecast_volatility.iloc[self.index_validation + 1:]

        self.fitted_conditional_variance = pd.concat([self.vol_pred_train, self.vol_pred_validation], axis=0)

    def set_return_train(self,
                         S):
        self.return_train = S
        self.r_true_train = self.return_train[self.ar_lags:self.index_validation]
        self.r_true_validation = self.return_train[self.index_validation + self.ar_lags + 1:]
        self.overwrite_params = self.fitted_model.params
        self.build_model()
        self.fit()


class ReturnModel:
    def __init__(self,
                 stock_model,
                 model_dict,
                 return_train,
                 index_validation=None):

        self.stock_model = stock_model
        self.model_dict = model_dict
        self.model_type = self.model_dict['type']
        self.return_train = return_train
        if index_validation is None:
            self.index_validation = len(self.return_train) - 1
        else:
            self.index_validation = index_validation - self.return_train.index[0]

        self.generator_train = None
        self.model = None
        self.ar_lags = self.model_dict['ar_lags']
        self.ma_lags = self.model_dict['ma_lags']

        self.fitted_model = None
        self.fitted_resids = None
        self.all_resids = None

        self.overwrite_params = None

        self.r_pred_train = None
        self.r_pred_validation = None
        self.r_true_train = self.return_train[self.ar_lags:self.index_validation]
        self.r_true_validation = self.return_train[self.index_validation + self.ar_lags + 1:]

    def build_model(self):
        if self.model_type == 'AR':
            self.model = arch.univariate.ARX(y=self.return_train.ravel(),
                                             lags=self.ar_lags,
                                             constant=False,
                                             rescale=False)
            # model_dict = {'mean': self.model_type,
            #               'vol': 'constant',
            #               'lags': self.ar_lags,
            #               'dist': 'ged'}
            # self.model = arch.arch_model(y=self.return_train.ravel(),
            #                              rescale=False,
            #                              **model_dict)

        elif self.model_type == 'NN':
            self.model_dict = {'layers': [10, 1]}
            self.model = Sequential()
            self.model.add(Dense(units=self.model_dict['layers'][0],
                                 activation='sigmoid',
                                 input_dim=self.ar_lags))
            #self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.model_dict['layers'][1],
                                 activation='linear'))
            self.model.compile(loss="mse",
                               optimizer="adam")

        elif self.model_type == 'RNN':
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
                params=self.fitted_model.params[:self.ar_lags]))

        elif self.model_type == 'NN':
            self.generator_train = TimeseriesGenerator(data=self.return_train.ravel(),
                                                       targets=self.return_train.ravel(),
                                                       length=self.ar_lags,
                                                       batch_size=8,
                                                       end_index=self.index_validation)
            self.model.fit_generator(self.generator_train,
                                     steps_per_epoch=1,
                                     epochs=200,
                                     verbose=0,
                                     use_multiprocessing=False)
            self.fitted_model = self.model

            y_pred_train = self.fitted_model.predict_generator(self.generator_train, verbose=0)

            self.fitted_resids = y_pred_train[:, 0] - self.return_train.iloc[self.ar_lags:self.index_validation]

    def evaluate(self):
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
                self.r_pred_validation = forecast.iloc[self.index_validation+1:]

            elif self.model_type == 'NN':
                self.r_pred_train = pd.Series(self.fitted_model.predict_generator(self.generator_train)[:, 0],
                                              index=self.return_train.index[self.ar_lags:self.index_validation])

                validation_generator = TimeseriesGenerator(data=self.return_train.ravel(),
                                                           targets=self.return_train.ravel(),
                                                           length=self.ar_lags,
                                                           start_index=self.index_validation)
                self.r_pred_validation = pd.Series(self.fitted_model.predict_generator(validation_generator)[:, 0],
                                                   index=self.return_train.index[self.index_validation+self.ar_lags+1:])

            self.all_resids = pd.concat([self.r_pred_train, self.r_pred_validation], axis=0)

    def get_return_pred_train(self):
        return self.r_pred_train

    def get_return_pred_validation(self):
        return self.r_pred_validation

    def summary(self):
        print(self.fitted_model.summary())

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

    def build_model(self):
        if self.model_type in ['arch', 'garch']:
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
                                 activation='sigmoid',
                                 input_dim=self.ar_lags))
            #self.model.add(Dropout(0.2))
            self.model.add(Dense(units=self.model_dict['layers'][1],
                                 activation='linear'))
            self.model.compile(loss="mse",
                               optimizer="adam")

    def fit(self):
        self.vol_true_train = self.resids.iloc[self.ar_lags:self.index_validation]
        self.vol_true_validation = self.resids.iloc[self.index_validation + self.ar_lags + 1:]

        if self.model_type in ['arch', 'garch']:
            self.fitted_model = self.model.fit(update_freq=5)

            self.fitted_conditional_volatility = self.fitted_model.conditional_volatility.dropna().apply(
                lambda var: var ** 0.5)

        elif self.model_type == 'NN':
            log_resids_sq = np.log(self.resids.apply(lambda resid: resid ** 2).values)
            if len(log_resids_sq.shape) > 1:
                log_resids_sq = log_resids_sq[:, 0]
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
                                     use_multiprocessing=False)
            self.fitted_model = self.model

            y_pred = np.exp(self.fitted_model.predict_generator(self.generator_train, verbose=0))
            self.fitted_conditional_volatility = pd.Series(y_pred[:, 0]).apply(lambda var: abs(var) ** 0.5)

    def evaluate(self):
        if self.vol_pred_train is None:
            if self.model_type == 'AR':
                self.vol_pred_train = self.fitted_model.forecast(end=self.index_validation,
                                                                 horizon=1) \
                    .volatility.dropna()
                self.vol_pred_validation = self.fitted_model.forecast(start=self.index_validation + 1,
                                                                      horizon=1) \
                    .volatility.dropna()

            elif self.model_type == 'NN':
                self.vol_pred_train = pd.Series(np.exp(self.fitted_model.predict_generator(self.generator_train)[:, 0]),
                                                index=self.resids.index[self.ar_lags:self.index_validation])

                self.vol_pred_validation = pd.Series(
                    np.exp(self.fitted_model.predict_generator(self.generator_validation)[:, 0]),
                    index=self.resids.index[
                          self.index_validation + self.ar_lags + 1:])

    def get_volatility_pred_train(self):
        return self.vol_pred_train

    def get_volatility_pred_validation(self):
        return self.vol_pred_validation

    def summary(self):
        print(self.fitted_model.summary())


def find_opt_params(stock,
                    model_type_to_change,
                    attribute_to_change,
                    value_list,
                    write_path='grid_search_params.xlsx'):
    test_dict = dict()
    for value in value_list:
        stock.set_model_attribute(model_type=model_type_to_change,
                                  attribute=attribute_to_change,
                                  value=value)
        stock.fit_model()
        stock.model.evaluate()
        stock.model.homoskedasticity_test()
        #stock.model.summary_fit()
        test_dict[value] = stock.get_metrics()
        stock.reset_metrics()

    metrics_list = list(test_dict[value_list[0]].keys())
    df_dict = {metric: [test_dict[key][metric] for key in value_list] for metric in metrics_list}
    pd.DataFrame.from_dict(df_dict).to_excel(write_path)


if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    data_path = 'clean_data.csv'
    return_column = 'log_return'

    model_dict = {
        'return_model': {
            'type': 'AR',
            'ar_lags': 1,
            'ma_lags': 5,
            'layers': [10, 1]
        },
        'volatility_model': {
            'type': 'NN',
            'ar_lags': 1,
            'ma_lags': 5,
            'layers': [10, 1]
        }
    }

    stock = Stock(stock_name='Bnp Paribas',
                  data_path=data_path,
                  return_column=return_column,
                  model_dict=model_dict)
    #
    # find_opt_params(stock=stock,
    #                 model_type_to_change='volatility_model',
    #                 attribute_to_change='ar_lags',
    #                 value_list=range(1, 11))

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

