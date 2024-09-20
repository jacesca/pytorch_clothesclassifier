import pandas as pd
import os
import numpy as np
import seaborn as sns
import pickle

from scipy.stats import pearsonr
from typing import Union, Literal, Tuple, Dict
from matplotlib import pyplot as plt
from dotenv import load_dotenv
from functools import partial
import matplotlib.patheffects as path_effects
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.impute import KNNImputer, SimpleImputer


load_dotenv()

SEED = int(os.getenv('SEED'))
SAVED_MODEL_PATH = os.getenv('SAVED_MODEL_PATH')
TEST_DATA_PATH = os.getenv('TEST_DATA_PATH')
FEATURES_SELECTED_PATH = os.getenv('FEATURES_SELECTED_PATH')
SCALER_PATH = os.getenv('SCALER_PATH')
PARQUET_FILE_PATH = os.getenv('PARQUET_FILE_PATH')
FEATURE_STORE_PATH = os.getenv('FEATURE_STORE_PATH')
SAVED_FIGURE_PATH = os.getenv('SAVED_FIGURE_PATH')
LOGS_PATH = os.getenv('LOGS_PATH')

CRED = '\033[42m'
CEND = '\033[0m'


def root_mean_squared_error(*args, **kwargs):
    return np.sqrt(mean_squared_error(*args, **kwargs))


def prepare_environment() -> None:
    """
    Set max columns to all, define size for matplotlib atributes
    and configure the randomnes with the SEED.
    """
    # Preparing the environment
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'axes.titlesize': 12, 'axes.titleweight': 'bold',
                         'axes.titlecolor': 'darkslategray',
                         'font.size': 10, 'figure.titlesize': 14,
                         'figure.titleweight': 'bold'})
    np.random.seed(SEED)


def hprint(msg: str) -> None:
    """Highlighted print"""
    print(CRED + msg + CEND)


def calculate_regressionmetrics(y_test: Union[list, np.ndarray,
                                              pd.DataFrame, pd.Series],
                                y_pred: Union[list, np.ndarray,
                                              pd.DataFrame, pd.Series]) \
                                                -> None:
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"""Evaluating the model:
    MSE : {mse}
    RMSE: {rmse}
    MAE : {mae}
    R²  : {r2}
    """)


def histogram_boxplot(feature: pd.Series,
                      bins: str = "auto",
                      figsize: Union[tuple, list] = (6, 3)) -> None:
    """ Boxplot and histogram combined
    feature: pandas.series
    bins: number of bins (default "auto")
    figsize: size of fig (default (6, 3))
    """
    mean = feature.mean()
    median = np.median(feature)
    min_v = feature.min()
    max_v = feature.max()

    sns.set(font_scale=.75)
    f, (ax_box, ax_hist) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid
        sharex=True,  # The X-axis will be shared among all the subplots
        gridspec_kw={"height_ratios": (.25, .75)},
        figsize=figsize
    )

    # Creating the subplots
    # Boxplot will be created and the mean value of the column will be
    # indicated using some symbol
    sns.boxplot(x=feature, ax=ax_box, color='violet',
                showmeans=True,
                meanprops={"marker": "o",
                           "markerfacecolor": "goldenrod",
                           "markeredgecolor": "silver",
                           "markersize": "10"})

    text = ax_box.annotate("Mean {:,.4f}".format(mean), fontsize='small',
                           xy=(mean, -0.15), color='g', weight='bold',
                           ha='center')
    text.set_path_effects([path_effects.Stroke(linewidth=3,
                                               foreground='black'),
                           path_effects.Normal()])
    ax_box.set_ylabel('BoxPlot\n')
    ax_box.set_xlabel('')

    # For histogram
    sns.histplot(x=feature, kde=False, bins=bins,
                 color="steelblue", ax=ax_hist)
    ax_hist.axvline(mean, color='g', linestyle='--')
    ax_hist.axvline(median, color='black', linestyle='-')

    min_max_pos = 0.05 * ax_hist.get_ylim()[1]
    text = ax_hist.annotate("Median {:,.4f}".format(median), fontsize='small',
                            xy=(median, ax_hist.get_ylim()[1]/2),
                            color='w', weight='bold', ha='center')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='k'),
                           path_effects.Normal()])
    text = ax_hist.annotate("Min {:,.4f}".format(min_v), fontsize='small',
                            xy=(min_v, min_max_pos),
                            color='w', weight='bold', ha='left')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='k'),
                           path_effects.Normal()])
    text = ax_hist.annotate("Max {:,.4f}".format(max_v), fontsize='small',
                            xy=(max_v, min_max_pos),
                            color='w', weight='bold', ha='right')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='k'),
                           path_effects.Normal()])

    ax_hist.set_ylabel('HistPlot')
    ax_hist.set_xlabel('')

    # Calculating the skewness
    # If the skewness is between -0.5 & 0.5, the data are nearly symmetrical.
    # If the skewness is between -1 & -0.5 (negative skewed) or between 0.5 &
    #     1 (positive skewed), the data are slightly skewed.
    # If the skewness is lower than -1 (negative skewed) or greater than 1
    #     (positive skewed), the data are extremely skewed.
    skewness = feature.skew()
    if skewness < -1:
        skewness_str = 'Extremely Negative Skewed'
    elif skewness < -0.5:
        skewness_str = 'Negative Skewed'
    elif skewness == 0:
        skewness_str = 'Simetrical Distributed'
    elif skewness <= 0.5:
        skewness_str = 'Nearly Simmetrical'
    elif skewness <= 1:
        skewness_str = 'Positive Skewed'
    elif skewness > 1:
        skewness_str = 'Extremely Positive Skewed'
    f.suptitle(f'EDA: {feature.name.upper()}\n'
               f'Skew: {skewness:0.4f} ({skewness_str})')
    plt.subplots_adjust(hspace=1, top=0.9)
    plt.tight_layout()
    f.savefig(f'{SAVED_FIGURE_PATH}eda_hist_{feature.name}.png')


def labeled_barplot(feature: pd.Series,
                    rotation: int = 0,
                    top: int = None,
                    title: str = None,
                    figsize: Union[list, tuple] = (6, 3),
                    order: bool = True) -> None:
    """Countplot of a categorical variable
    feature: pandas.series
    rotation: rotation of xticks (default 0)
    top: max rows to return. If none is provided all rows are returned.
        (Default: None)
    title: title of the plot. If none value is provided, feature names are
        displayed. (Default: None)
    figsize: size of fig (default (6, 3))
    """
    title = title if title else f'EDA: {feature.name}'

    sns.set(font_scale=.75)
    fig = plt.figure(figsize=figsize)

    # Convert the column to a categorical data type
    feature = feature.astype('category')
    origin = feature.copy()

    labels = feature.value_counts().index
    if top:
        labels = labels[:top]
        feature = feature.loc[feature.isin(labels)]

    ax = sns.countplot(x=feature, hue=feature, palette='Paired',
                       order=labels if order else None)
    ax.set_xlabel('')

    # custom label calculates percent and add an empty string so 0 value bars
    # don't have a number
    for container in ax.containers:
        labels = [f'{h:.0f}\n( {h/origin.count()*100:0.1f}% )'
                  if (h := v.get_height()) > 0 else '' for v in container]
        ax.bar_label(container, labels=labels, label_type='edge',
                     # color='white', label_type='center'
                     fontsize='small', weight='bold')
    ylim = plt.ylim()
    plt.ylim(ylim[0], ylim[1]*1.1)

    plt.suptitle(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    # plt.autoscale(enable=True, axis='y', tight=False)
    fig.savefig(f'{SAVED_FIGURE_PATH}eda_bar_{feature.name}.png')


def impute_missing_values_with_KNN(df: pd.DataFrame,
                                   n_neighbors: int = 2,
                                   weights: Literal['uniform', 'distance'] = 'uniform') \
                                    -> Tuple[pd.DataFrame, KNNImputer]:  # noqa
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    df_imputed = imputer.fit_transform(df)
    return df_imputed, imputer


def impute_missing_values(df: pd.DataFrame,
                          features: list,
                          strategy: Literal['mean', 'median',
                                            'most_frequent',
                                            'constant'] = 'mean',
                          constant: Union[str, int, float] = None) \
                            -> Tuple[pd.DataFrame, Dict[str, SimpleImputer]]:
    df_imputed = df.copy()
    imputers = {}
    for col in features:
        imputer = SimpleImputer(strategy='mean')
        df_imputed[[col]] = imputer.fit_transform(df_imputed[[col]])
        imputers[col] = imputer
    return df_imputed, imputers


def end_environment() -> None:
    plt.show()
    plt.style.use('default')


def save_object(obj, filename):
    """To save objects that can be loaded later"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def population_stability_index(expected: Union[pd.DataFrame, pd.Series,
                                               np.ndarray, list],
                               actual: Union[pd.DataFrame, pd.Series,
                                             np.ndarray, list],
                               buckettype: Literal['bins',
                                                   'quantiles'] = 'bins',
                               buckets: int = 10,
                               axis: Literal[0, 1] = 0,
                               show_graph: bool = True,
                               feature_name: str = 'Feature'):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values
       buckettype: type of strategy for creating buckets, bins splits into
                   even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for
             horizontal
       show_graph: bool, if True a graph of the expected and actual values
                   is shown.

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       mwburke.github.io.com

    Source:
        https://github.com/mwburke/population-stability-index
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array),
                                      np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b)
                                    for b in breakpoints])

        expected_fractions = np.histogram(expected_array,
                                          breakpoints)[0] / len(expected_array)
        actual_fractions = np.histogram(actual_array,
                                        breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value

        psi_value = sum(sub_psi(expected_fractions[i], actual_fractions[i])
                        for i in range(0, len(expected_fractions)))

        return psi_value

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[1 - axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

    if psi_values < 0.1:
        explanation = 'Very low change, considered stable.'  # noqa
    elif 0.1 <= psi_values < 0.25:
        explanation = 'Moderate change, considered some adjustments.'  # noqa
    else:  # psi_value >= 0.25:
        explanation = 'High change, considered significant.'  # noqa

    if show_graph:
        fig, ax = plt.subplots()
        sns.kdeplot(expected, fill=True, ax=ax)
        sns.kdeplot(actual, fill=True, ax=ax)
        ax.set(yticklabels=[], xticklabels=[])
        sns.despine(left=True)
        fig.suptitle(f'Population Stability Index (PSI): {feature_name} - {psi_values:.4f}')  # noqa
        ax.set_title(explanation)
        ax.axis('off')
        plt.legend(['Expected', 'Actual'])
        fig.savefig(f'{SAVED_FIGURE_PATH}data_drift_{feature_name.lower()}_psi.png')  # noqa

    return psi_values, explanation


def load_object(filename):
    """Get back the saved object"""
    with open(filename, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


def linear_model_plot(x: str, y: str, df: pd.DataFrame) -> None:
    '''Plot a linear model between x and y'''

    def coorrfunc(x, y, **kws):
        (r, p) = pearsonr(x, y)
        return r ** 2

    g = sns.jointplot(x=x, y=y, data=df, kind="reg")
    g.ax_joint.text(s=f"r² = {coorrfunc(df[x], df[y])}",
                    x=.05, y=.95, transform=g.ax_joint.transAxes,
                    bbox={'boxstyle': 'round', 'pad': 0.25,
                          'facecolor': 'white', 'edgecolor': 'gray'})


print = partial(print, sep='\n', end='\n\n')
