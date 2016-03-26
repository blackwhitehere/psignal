__author__ = 'cparlin'
__author__ = 'stan'
import pandas as pd
import numpy as np
import os
import Quandl as q
import statsmodels.tsa.stattools as stat
import statsmodels.api as sm
from config import config as c
from statsmodels.stats.stattools import durbin_watson


def get_clustered_data(path):
    df = pd.read_csv(filepath_or_buffer=path, delimiter=",", header=0)
    df = df.iloc[:, 1:]
    df['TIMESTAMP_UTC'] = pd.to_datetime(arg=df.TIMESTAMP_UTC, infer_datetime_format='true', unit='D')
    return df


def get_cluster_data_with_CSI(path):
    df = pd.read_csv(filepath_or_buffer=path, delimiter=",", header=0)
    df = df.iloc[:, 1:]
    return df


def get_raw_data_col_names():
    return ['sym', 'date', 'sen', 'cluster']


def get_stock_returns(db, sym, column, start, end, api_token):
    try:
        rs = q.get(dataset=(db + '/' + sym + "." + column), trim_start=start, trim_end=end, transformation='rdiff', \
                   authtoken=api_token)
        return rs
    except:
        return pd.DataFrame({})


def resample(ts, periods):
    return ts.resample(periods, how=cumret)


def dw_stat(ts):
    return durbin_watson(ts)


def aug_dickey_fuller(ts, reg_type=c.df_reg_type, alpha=c.alpha):
    dft = stat.adfuller(ts, maxlag=2, regression=reg_type, autolag="BIC", store=True, regresults=True)
    pvalue = dft[1]
    return pvalue <= alpha


def take_difference(ts):
    if type(ts) is np.ndarray:
        delta = t_plus_n(ts, 1) - t_plus_n_minus_1(ts, 0)
        index = np.arange(len(delta))
    else:
        delta = t_plus_n(ts, 1).values - t_plus_n_minus_1(ts, 0).values
        index = ts.index[1:len(ts)]
    ret = pd.Series(delta.ravel(), index=index)
    return ret


def integration_degree(ts, max_degree=c.max_degree, reg_type=c.df_reg_type, alpha=c.alpha):
    df_passed = False
    iteration = 0
    while (df_passed is False) & (iteration <= max_degree):
        dft = aug_dickey_fuller(ts, reg_type, alpha)
        if dft:
            df_passed = True
        else:
            ts = take_difference(ts)
            iteration += 1
    return iteration


def detrend(degree, ts):
    for i in range(0, degree):
        ts = take_difference(ts)
    return ts


def t_plus_n_minus_1(ts, n):
    return ts[n:(len(ts) - 1)]


def t_plus_n_minus_z(ts, n, z):
    return ts[n:(len(ts) - z)]


def t_plus_n(ts, n):
    return ts[n:len(ts)]


def market_model(stock_ts, market_ts):
    r_t_plus_one = t_plus_n(stock_ts, 1)
    # r_t = t_plus_n_minus_1(stock_ts, 0)
    m_t = t_plus_n_minus_1(market_ts, 0)
    dep_vars = m_t.values
    dep_vars = sm.add_constant(dep_vars)
    linreg = sm.GLS(r_t_plus_one.values, dep_vars)
    return linreg.fit()


def sentiment_model(market_model_resid, stock_ts, sentiment_ts):
    prevresids = t_plus_n_minus_1(market_model_resid, 0)
    r_t = t_plus_n_minus_z(stock_ts, 1, 1)
    si_t = t_plus_n_minus_z(sentiment_ts, 1, 1)
    print(len(prevresids))
    #print(prevresids)
    print(len(r_t))
    print(len(si_t))
    dep_vars = pd.DataFrame(np.concatenate(
        (prevresids.reshape(len(prevresids), 1), r_t.values.reshape(len(r_t), 1), si_t.values.reshape(len(si_t), 1)),
        axis=1))
    dep_vars.columns = ['Residuals_t-1', 'Returns_t-1', 'CSI_t-1']
    y = t_plus_n(market_model_resid, 1)
    print(len(y))
    dep_vars.reset_index(inplace=True)
    y = pd.DataFrame(y, index=range(0, len(y)))
    dep_vars = sm.add_constant(dep_vars)
    linreg = sm.GLS(y, dep_vars)
    return linreg.fit()


def run_model(list_of_ts, model):
    degrees = [integration_degree(ts) for ts in list_of_ts]
    max_degree = max(degrees)
    integrated_factors = [detrend(max_degree, ts) for ts in list_of_ts]
    return model(*integrated_factors), max_degree


def change_in_sentiment_model(sentiment_ts):
    delta = (t_plus_n(sentiment_ts, 1).values - t_plus_n_minus_1(sentiment_ts, 0).values) / t_plus_n_minus_1(
        sentiment_ts, 0).values
    index = sentiment_ts.index[1:len(sentiment_ts)]
    percentage_change_in_sentiment = pd.Series(delta.ravel(), index=index)
    return percentage_change_in_sentiment


def format_result(param, pvalue):
    return str(param) + " ({0})".format(pvalue)


def mean_si(df):
    gdf = df.groupby('TIMESTAMP_UTC', 'ClusterID').mean()


def cumret(ts):
    ts = np.add(ts, 1)
    ts = np.cumprod(ts)[len(ts) - 1]
    ts = ts - 1
    return ts


def volatility(ts):
    ts = ts.var()
    return ts
