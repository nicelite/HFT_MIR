import pandas as pd
import arch
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv('clean_data.csv')

df_small = df.iloc[1:10000]
df_small.set_index('date', inplace=True)

# Data analysis
# TODO : Mean, variance, skewness, kurtosis

# Correlograme
plot_acf(df_small['log_return'])
# TODO: correlogramme r, r^2

# Test stationarite (Dickey Fuller)
# Test hhomosckedasticite

# analyse residus par modèle ar

model = arch.univariate.ARX(y=df_small['log_return'],
                            lags=2,
                            rescale=True)
res = model.fit(disp='off')

print(res.summary())
plot_acf(res.resid)
res.plot()
#pd.plot(res.resid, kind='hist')