import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import os
from datetime import datetime
import seaborn as sns
# from darts.models.forecasting import arima
import statsmodels.api as sm
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
from scipy import optimize
from scipy.stats import norm
from math import sqrt
import joblib
from joblib import dump, load

import warnings
warnings.filterwarnings('ignore')
from warnings import catch_warnings

palette = sns.color_palette("mako_r", 6)


## search for the best model
# P <= 3, Q <= 1; p <= 3, q <= 27 -> impossible to solve for such a range. so truncate.
# Note large q won't work. Computationally expensive.

# modified utils from the following sources
# https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
# https://www.kaggle.com/code/leandrovrabelo/climate-change-forecast-sarima-model/notebook

order_aic = [] # store AIC per model here. Complementary to CV
## currently not in use

# forecast function
def sarima_forecast(history, config, aic = True):
    """
    order = (p,d,q)
    sorder = (P,D,Q,s)
    """
    global order_aic
    order, sorder = config[:3], config[3:]
    # define model
    model = sarimax.SARIMAX(history, order=order,
                          seasonal_order=(0,0,0,0),
                          trend=None, enforce_stationarity=False,
                          enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False) # do not print convergence message
    # if aic:
    #   order_aic = order_aic + [(config, model_fit.aic)]
    #   print(f'Model {config}: {model_fit.aic}')
    # make one step forecast
    yhat = model_fit.predict(start = len(history), end = len(history))
    return yhat[-1]


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

def walk_forward_validation(train, val, col, n_val, cfg):
        predictions = np.zeros(len(val))
        # seed history with training dataset
        history = list(train[col])
        val = list(val[col])
        # step over each time-step in the test set
        for i in tqdm(range(len(val))):
        # fit model and make forecast for history
            yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
            predictions[i] = yhat
        # add actual observation to history for the next loop
            history.append(val[i])
        # estimate prediction error
        error = measure_rmse(val, list(predictions))
        return error, predictions


# score a model, return None on failure
def score_model(train, val, col, n_val, cfg, debug=False, save = True):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result, preds = walk_forward_validation(train, val, col, n_val, cfg)
    else:
    #   try:
    # # never show warnings when grid searching, too noisy
    #       with catch_warnings():
    #         warnings.filterwarnings("ignore")
        result, preds = walk_forward_validation(train, val, col, n_val, cfg)
      # except:
      #     error = None
    # check for an interesting result
    if result is not None:
      print(' > Model[%s] %.4f' % (key, result))
    if save:
        np.save('~/arma_{key}_predictions.npy', predictions)
    return (key, result)


# we need to set to sarimax gridsearch
def grid_search(train, val, col, n_val, cfg_list, parallel=True):
    global order_aic
    scores = None
    if parallel:
    # execute configs in parallel
        executor = Parallel(n_jobs=-1, backend='multiprocessing')
        tasks = (delayed(score_model)(train, val, col, n_val, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(train, val, col, n_val, cfg) for cfg in cfg_list]
        # remove empty results
        scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda x: x[1])
    return scores


## Fourier Transform from lab_10
def calculate_fourier_series(dt: float, signal: np.ndarray):
    """Calculate the Fourier series coefficients using the Fourier transform.

    Args:
        dt: Interval between samples.
        signal: Real signal at each point in time.

    Returns:
        Frequencies, A_n, and B_n coefficients for the fourier series
        representaiton.

    Notes:
        Take advantage of numpy.fft. Remember that the signal is real not
        complex. You may want to take advantage of the norm keyword.
    """
    # Placeholders.
    a_n = None
    b_n = None

    # TODO: Calculate the frequencies of each A_n and B_n. Remember that the
    # maximum frequency you can measure (and the maximum value numpy.fft will return
    # for real-valued signals) will be the Nyquist frequency.
    frequencies = np.abs(np.fft.fftfreq(n = len(signal), d = dt)[:len(signal)//2 + 1])

    # TODO: Calculate the fourier series coefficients. Compare the equations from
    # the notes, and read the numpy.fft documentation carefully.
    fourier_transform = np.fft.fft(signal, n = len(signal), norm = 'forward')[:len(signal)//2 + 1] # \tilde x_n = c_n
    a_n = 2*fourier_transform.real
    b_n = -2*fourier_transform.imag

    return frequencies, a_n, b_n


def reconstructed_signal(frequency_mask: float, signal: np.ndarray):
    """Return the signal with a mask applied to the Fourier series.

    Args:
        frequency_mask: Terms in the Fourier series to mask.
        signal: Real signal at each point in time.

    Returns:
        Reconstructed signal after frequency mask has been applied.

    Notes:
        Take advantage of numpy.fft. Remember that the signal is real not
        complex.
    """
    # TODO: Calculate the fourier transform, apply the mask,
    # and reverse the transformation.
    fourier_transform = np.fft.fft(signal, n = len(signal), norm = 'forward')
    return np.fft.ifft(fourier_transform * frequency_mask, norm = 'forward').real


## naive version
def plot_triple(df, n_lags_ar, n_lags_ma, acf_sm, pacf_sm, col, fig_name = ''):
    acf_error_estimate = 2 * np.ones(n_lags_ar) / np.sqrt(len(df))
    pacf_error_estimate = 2 * np.ones(n_lags_ma) / np.sqrt(len(df))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(22, 15), dpi=200)
    fontsize = 20
    labelsize = 12
    colors = ['#7fcdbb', '#2c7fb8']
    palette = sns.color_palette("mako_r", 6)

    years_txt, years = np.unique(df['Time'].dt.year, return_index=True)

    ax1.set_title('Data', fontsize=fontsize)
    sns.lineplot(x=df.index, y=df[col],
                 color=palette[1], label= col, alpha=0.9, ax=ax1)
    ax1.set_xticklabels([])
    ax1.set_xticks(years[1:])
    ax1.set_xticklabels(years_txt[1:], rotation=45, ha='right')
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=fontsize)
    ax1.set_xlabel('Year', fontsize=fontsize, labelpad=5.0)
    ax1.legend(fontsize=12)

    ax2.stem(acf_sm, linefmt=colors[0], markerfmt=colors[0], basefmt='k', label='Estimated ACF')
    ax2.fill_between(np.arange(1, n_lags_ar + 1), acf_error_estimate, -acf_error_estimate,
                     color=colors[1], label='Error in Estimate', alpha=0.2)
    ax2.set_ylim([-1.2, 1.2])
    ax2.set_title('ACF Estimate', fontsize=fontsize)
    ax2.set_xlabel(r'$|h|$', fontsize=fontsize)
    ax2.set_ylabel(r'$\rho(|h|)$', fontsize=fontsize)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend(fontsize=fontsize)

    ax3.stem(pacf_sm, linefmt=colors[0], markerfmt=colors[0], basefmt='k', label='Estimated PACF')
    ax3.fill_between(np.arange(1, n_lags_ma + 1), pacf_error_estimate, -pacf_error_estimate,
                     color=colors[1], label='Error in Estimate', alpha=0.2)
    ax3.set_ylim([-1.0, 1.2])
    ax3.set_title('PACF Estimate', fontsize=fontsize)
    ax3.set_xlabel(r'$|h|$', fontsize=fontsize)
    ax3.set_ylabel(r'$\mathrm{PACF}(|h|)$', fontsize=fontsize)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.legend(fontsize=fontsize)

    plt.tight_layout()
    plt.show()
    fig.savefig(f'{fig_name}.png')


   ## built-in plot_pcaf, plot_acf
def plot_triple_built_in(df, n_lags_ar, n_lags_ma, seasonality, lag_ma, col, fig_name = ''):
    acf_error_estimate = 2 * np.ones(n_lags_ar) / np.sqrt(len(df))
    pacf_error_estimate = 2 * np.ones(n_lags_ma) / np.sqrt(len(df))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(22, 15), dpi=200)
    fontsize = 20
    labelsize = 12
    colors = ['#7fcdbb', '#2c7fb8']
    palette = sns.color_palette("mako_r", 6)

    years_txt, years = np.unique(df['Time'].dt.year, return_index=True)

    ax1.set_title('Data', fontsize=fontsize)
    sns.lineplot(x=df.index, y=df[col],
                 color=palette[1], label= col, alpha=0.9, ax=ax1)
    ax1.set_xticklabels([])
    ax1.set_xticks(years[1:])
    ax1.set_xticklabels(years_txt[1:], rotation=45, ha='right')
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=fontsize)
    ax1.set_xlabel('Year', fontsize=fontsize, labelpad=5.0)
    ax1.legend(fontsize=12)


    plot_acf(df[col], lags = np.arange(n_lags_ar)[::seasonality], ax = ax2)
    ax2.set_ylim([-1.2, 1.2])
    ax2.set_title('ACF Estimate', fontsize=fontsize)
    ax2.set_xlabel(r'$|h|$', fontsize=fontsize)
    ax2.set_ylabel(r'$\rho(|h|)$', fontsize=fontsize)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    # ax2.legend(fontsize=fontsize)

    plot_pacf(df[col], lags = np.arange(n_lags_ma)[::lag_ma], ax = ax3)
    ax3.set_ylim([-1.0, 1.2])
    ax3.set_title('PACF Estimate', fontsize=fontsize)
    ax3.set_xlabel(r'$|h|$', fontsize=fontsize)
    ax3.set_ylabel(r'$\mathrm{PACF}(|h|)$', fontsize=fontsize)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    # ax3.legend(fontsize=fontsize)

    plt.tight_layout()
    plt.show()
    fig.savefig(f'{fig_name}.png')




    ## forecast plot
def plot_forecast(X_train, X_val, preds, n_from_train_end, n_from_val_start, col, pred_col, save = False, label = ''):
    fig, ax = plt.subplots(1, 1, figsize=(22, 5), dpi=200)
    fontsize = 20
    labelsize = 12
    colors = ['#7fcdbb', '#2c7fb8']
    palette = sns.color_palette("mako_r", 6)

    X_train_trunc = X_train[-n_from_train_end:]
    X_val_trunc = X_val[:n_from_val_start]
    df = pd.concat([X_train_trunc, X_val_trunc], axis = 0)
    X_pred = pd.DataFrame(preds.squeeze(1)[:n_from_val_start], columns = [pred_col], index = X_val_trunc.index)
    X_pred['Time'] = X_val_trunc.Time
    years_txt, years = np.unique(df['Time'].dt.year, return_index=True)

    ax.set_title('Forecast', fontsize=fontsize)
    sns.lineplot(x=X_train_trunc.index, y=X_train_trunc[col],
                 color=palette[1], lw = 1, label= f'Train_{col}', alpha=0.9, ax=ax)
    sns.lineplot(x=X_val_trunc.index, y=X_val_trunc[col],
                 color=palette[1], lw = 1, label= f'Val_{col}', alpha=0.9, ax=ax)
    sns.lineplot(x=X_pred.index, y=X_pred[pred_col],
                 color=palette[-1], lw = 1, label= f'Pred_{pred_col}', alpha=1, ax=ax)
    ax.set_xticklabels([])
    ax.set_xticks(years[1:])
    ax.set_xticklabels(years_txt[1:], rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=fontsize)
    ax.set_xlabel('Year', fontsize=fontsize, labelpad=5.0)
    ax.legend(fontsize=12)
    if save:
        fig.savefig(f'pred_{n_from_train_end}_{n_from_val_start}.png', bbox_inches = 'tight')

        

## modified, somewhat hard-coded ...
## used for error analysis at the end
def plot_error(df, figsize=(20,8)):
    '''
    There must have 3 columns following this order: Temperature, Prediction, Error
    '''
    colors = ['#7fcdbb', '#2c7fb8']
    palette = sns.color_palette("mako_r", 6)
    fontsize = 10

    plt.figure(figsize=figsize)
    plt.suptitle('Error Analysis', fontsize = 20, y = 1.0)

    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax4 = plt.subplot2grid((2,2), (1,1))
    
    years_txt, years = np.unique(df['Time'].dt.year, return_index=True)
    #Plotting the Current and Predicted values
    sns.lineplot(x= df.index, y= df['T_AVG'],
                 color=palette[1], lw = 1, label= 'val', alpha=0.5, ax=ax1)
    sns.lineplot(x= df.index, y= df['preds'],
                 color=palette[-1], lw = 1, label= 'pred', alpha=1.0, ax=ax1)

    ax1.set_xticklabels([])
    ax1.set_xticks(years)
    ax1.set_xticklabels(years_txt, rotation=45, ha='right', fontsize = fontsize)
    ax1.set_xlabel('Year', fontsize = 15)
    ax1.set_ylabel('T_AVG', fontsize = 15)
    ax1.set_title('Current and Predicted Values', fontsize = 18, y = 1.05)
    
    # Residual vs Predicted values
    ax2.scatter(df.preds, df.error, color = palette[0], alpha = 0.7)
    ax2.set_xlabel('Predicted Values', fontsize = 15)
    ax2.set_ylabel('Errors', fontsize = 15)
    ax2.set_title('Errors versus Predicted Values', fontsize = 18, y = 1.05)
    
    ## QQ Plot of the residual
    qqplot = sm.qqplot(data = df.error, dist = norm, loc = df.error.mean(),
                            scale = df.error.std(),
                           markerfacecolor= palette[0], markeredgecolor= 'None',
                           markersize = 5.5, alpha = 0.3, ax = ax3)
    sm.qqline(line='45', color = palette[-1], lw = 1.5, ax = ax3)
    ax3.set_xlabel('Theoretical Quantiles', fontsize = 15)
    ax3.set_ylabel('Sample Quantiles', fontsize = 15)
    ax3.set_title('QQ plot', fontsize = 18, y = 1.05)
    
    # Autocorrelation plot of the residual
    plot_acf(df.error, lags= 50, zero=False, ax=ax4)
    ax4.set_ylim([-0.5, 0.5])
    ax4.set_title('ACF Estimate', fontsize=18, y = 1.05)
    ax4.set_xlabel(r'$|h|$', fontsize=15)
    ax4.set_ylabel(r'$\rho(|h|)$', fontsize=15)
    ax4.set_xticklabels([f'{int(i)}' for i in ax4.get_xticks()], rotation=45, ha='right', fontsize = fontsize)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', bbox_inches = 'tight')
    plt.show()
    