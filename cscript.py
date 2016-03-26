__author__ = 'cparlin'
__author__ = 'stan'

import cutility as u
import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt
from config import config as c

# bmth_us = CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())

csi_clusters_file_dict = c.csi_clusters_file_dict


def reg_clusters_with_CSI(file):
    file_path = c.clusters_folder_with_CSI + file
    twt = u.get_cluster_data_with_CSI(file_path)
    print(twt.head())

    stocks_to_check = twt['SYMBOL'].unique()
    print(stocks_to_check)
    # np.savetxt(c.reg_results_folder + 'Stocks_in_the_network_based_on_' + file, stocks_to_check)

    mi = u.get_stock_returns(c.market_return_db, c.market_returns_index, c.market_returns_column, c.start, c.end,
                             c.api_token)

    custom_mi = mi.resample(c.period, how=u.cumret, label='right')
    custom_mi = custom_mi.loc[custom_mi.index > c.actual_start]
    print(custom_mi)

    results_dict = {}
    result_df = pd.DataFrame(columns=[  'Stock',
                                        'Mean sentiment model on returns',
                                        'Degree of Integration market model',
                                        'Durbin Watson Statistic market model',
                                        'Degree of Integration sentiment model',
                                        'Durbin Watson Statistic sentiment model',
                                        # 'Percentage change in sentiment model on returns',
                                        'Mean sentiment model on volatility',
                                        'Degree of Integration market model',
                                        'Durbin Watson Statistic market model',
                                        'Degree of Integration sentiment model',
                                        'Durbin Watson Statistic sentiment model'
                                        # 'Percentage change in sentiment model on volatility'
                                    ]
                                 )

    for stock in stocks_to_check.tolist():
        print(stock)
        stock_returns = u.get_stock_returns(c.stock_returns_db, stock, c.stock_returns_col, c.start, c.end, c.api_token)

        if not stock_returns.empty:
            print('Downloaded stock data for ' + stock)
            stats_to_save = []
            stats_to_save.append(stock)

            # Resample stock returns
            stock_r = stock_returns.resample(c.period, how=u.cumret, label='right')
            stock_v = stock_returns.resample(c.period, how=u.volatility, label='right')
            # in current setup this creates 30 bi-monthly periods from 01-2011-->02-2011 to 10-2015-->12-2015:
            stock_r = stock_r.loc[stock_r.index > c.actual_start]
            stock_v = stock_v.loc[stock_v.index > c.actual_start]
            if len(stock_r) < 15:
                print("Stock " + stock + " has too few observations to be included in analysis")
                break

            # Name timeseries
            mask = (twt['SYMBOL'] == stock)
            stock_period_clusters = twt[mask]

            stock_sentiment_ts = stock_period_clusters['mean']
            delta_sentiment = u.change_in_sentiment_model(stock_sentiment_ts)
            stock_r_ts = stock_r[c.returns_column_name]
            stock_v_ts = stock_v[c.returns_column_name]
            market_index_ts = custom_mi['Adjusted Close']

            # Market model on returns
            def run_models(stock_ts, sentiment_ts, market_ts):
                mm_list_of_factors = [stock_ts, market_ts]
                market_model_reg, mm_degree = u.run_model(mm_list_of_factors, u.market_model)

                r_ts = u.detrend(mm_degree, stock_ts)
                si_ts = u.detrend(mm_degree, sentiment_ts)
                sm_list_of_factors = [market_model_reg.resid, r_ts, si_ts]
                print([len(x) for x in sm_list_of_factors])
                print('')
                #print(market_model_reg.resid)
                sentiment_model_reg, sm_degree = u.run_model(sm_list_of_factors, u.sentiment_model)

                models_dict = {'Market_model': [market_model_reg, mm_degree],
                               'Sentiment_model': [sentiment_model_reg, sm_degree],
                               }
                # save summary stats to a list
                cell = u.format_result(sentiment_model_reg.params[2], sentiment_model_reg.pvalues[2])
                stats_to_save.append(cell)
                stats_to_save.append(mm_degree)
                stats_to_save.append(u.dw_stat(market_model_reg.resid))
                stats_to_save.append(sm_degree)
                stats_to_save.append(u.dw_stat(sentiment_model_reg.resid))
                return models_dict

            md1 = run_models(stock_r_ts, stock_sentiment_ts, market_index_ts)
            #md2 = run_models(stock_r_ts, delta_sentiment, market_index_ts)
            md3 = run_models(stock_v_ts, stock_sentiment_ts, market_index_ts)
            #md4 = run_models(stock_v_ts, delta_sentiment, market_index_ts)

            models_dict = {**md1,
                           #**md2,
                           **md3,
                           #**md4
                           }

            results_dict[stock] = {'returns_ts': stock_r_ts,
                                   'volatility_ts': stock_v_ts,
                                   'market_ts': market_index_ts,
                                   'sentiment_ts': stock_sentiment_ts,
                                   'dict_of_models': models_dict}

            print(stats_to_save)
            result_df.loc[len(result_df)] = [x for x in stats_to_save]

        else:
            print('Returns of stock ' + stock + ' were not found in the ' + c.stock_returns_db + ' database')

    return result_df, results_dict


for key, file in csi_clusters_file_dict.items(): #{'Bearish Intensity': 'BEARISH_INTENSITY.csv'}.items():
    df_result, dict_of_results = reg_clusters_with_CSI(file)
    print(df_result)
    df_result.to_csv(c.reg_results_folder + key + '_results.csv', index=False)

print("Done")
