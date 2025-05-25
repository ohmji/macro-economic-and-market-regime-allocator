# General
import pandas as pd
import numpy as np
import itertools
import os

# Visualization
import plotly.express as px
from src.data_understanding import plot_regimes
from src.model_evaluation import plot_feature_importance

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Model Evaluation
from src.model_evaluation import plot_confusion_matrix, evaluation_metrics, plot_feature_importance

import ipywidgets as widgets
from ipywidgets import interact


def load_market_regimes_data(path='./data/datasets/mkt_regs_ds.csv', target_col='MktRegime', split_date='1973-01-01', horizon=1):
    df = pd.read_csv(path)
    feature_cols = df.columns.drop([target_col, 'Date'])
    df[feature_cols] = df[feature_cols].shift(horizon)
    df.dropna(inplace=True)
    df_train = df[df['Date'] < split_date]
    df_oos = df[df['Date'] >= split_date]
    return df, df_train, df_oos, feature_cols


def plot_class_imbalance(df_oos, target_col):
    target_counts = df_oos[target_col].value_counts()
    target_counts.index = ['Normal', 'Crash']
    fig = px.pie(
        values=target_counts,
        names=target_counts.index)
    fig.update_traces(textinfo='percent', pull=[0.05, 0.05], textfont_size=14)
    fig.update_layout(title_text='<b> Class Imbalance in Out of Sample Data </b>', title_x=0.5,
                      autosize=False, width=400, height=400, showlegend=True)
    fig.show()


def cross_validate_models(df_train, feature_cols, target_col):
    model_dict_cv = {
        ('DT', DecisionTreeClassifier): {
            'max_depth': [3, 5, 8, 10],
            'splitter': ['best', 'random'],
            'min_samples_split': [2, 3, 5]
        },
        ('RF', RandomForestClassifier): {
            'random_state': [42],
            'max_depth': [3, 5, 8, 10],
            'n_estimators': [100, 200, 400]
        },
        ('XGB', xgb.XGBClassifier): {
            'booster': ['gbtree'],
            'max_depth': [3, 5, 8, 10],
            'n_estimators': [100, 200, 400],
            'random_state': [42],
            'objective': ['binary:logistic']
        }
    }
    best_params_dict = {}
    for model_tuple, param_grid in model_dict_cv.items():
        all_grid = list(dict(zip(param_grid, x)) for x in itertools.product(*param_grid.values()))
        cv_scores = []
        for param in all_grid:
            tscv = TimeSeriesSplit(n_splits=3)
            model = model_tuple[1](**param)
            score = []
            for train_index, test_index in tscv.split(df_train):
                X_train, X_test = df_train[feature_cols].iloc[train_index], df_train[feature_cols].iloc[test_index]
                y_train, y_test = df_train[target_col].iloc[train_index], df_train[target_col].iloc[test_index]
                model.fit(X_train, y_train)
                y_binary = model.predict(X_test)
                score.append(np.mean(y_binary == y_test))
            cv_scores.append(np.mean(score))
        best_params = all_grid[np.argmax(cv_scores)]
        best_params_dict[model_tuple] = best_params
    return best_params_dict


def rolling_oos_predictions(df, best_params_dict, feature_cols, target_col, split_date, roll_window=150):
    oos_start_index = df[df['Date'] == split_date].index[0]
    df_pred = df.iloc[oos_start_index - roll_window:, :]
    X = df_pred[feature_cols]
    y = df_pred[target_col]
    date_range = df_pred['Date']
    model_counter = 0
    for model_tuple, param in best_params_dict.items():
        y_prob = np.array([])
        date = np.array([], dtype='datetime64[s]')
        y_actual = np.array([])
        for i in np.arange(0, len(df_pred) - roll_window):
            model = model_tuple[1](**param)
            X_fit = X.iloc[i: i + roll_window, :]
            y_fit = y.iloc[i: i + roll_window]
            model = model.fit(X_fit, y_fit)
            X_predict = X.iloc[i + roll_window: i + roll_window + 1, :]
            y_pred = model.predict_proba(X_predict)[:, 1]
            y_prob = np.hstack((y_prob, y_pred))
            date = np.hstack((date, date_range.iloc[i + roll_window: i + roll_window + 1].values))
            y_actual = np.hstack((y_actual, y.iloc[i + roll_window: i + roll_window + 1].values))
        if model_counter == 0:
            res_rolling_all = pd.DataFrame.from_dict({'Date': date,
                                                      'Regime': y_actual,
                                                      model_tuple[0]: y_prob})
            model_counter += 1
        else:
            res_rolling_all[model_tuple[0]] = y_prob
    return res_rolling_all


def plot_model_results(res_rolling_all):
    report_dir = './data/market'
    os.makedirs(report_dir, exist_ok=True)
    for model in ['DT', 'RF', 'XGB']:
        fig = plot_regimes(
            date_series=res_rolling_all['Date'],
            data_series=res_rolling_all[model],
            regime_series=res_rolling_all['Regime'],
            area_name='Crash',
            data_name=model,
            x_axis_title='Date',
            y_axis_title='Crash Probability',
            plot_title=f'Probability of Crash Predicted by {model}',
            log_scale=False
        )
        fig.write_image(f'{report_dir}/{model}_crash_probs.png')
        fig = plot_confusion_matrix(
            y_true=res_rolling_all['Regime'].values,
            y_pred=(res_rolling_all[model] > 0.5).astype(float).values,
            class_names=['Normal', 'Crash'],
            title=f'{model} Confusion Matrix'
        )
        fig.write_image(f'{report_dir}/{model}_confusion_matrix.png')
    metrics_result, fig = evaluation_metrics(
        pred_df=res_rolling_all,
        model_cols=['DT', 'RF', 'XGB'],
        y_true_col='Regime'
    )
    return metrics_result


def save_model_evaluation_reports(res_rolling_all, metrics_result, h=1, report_dir='./data/report/market'):
    os.makedirs(report_dir, exist_ok=True)
    metrics_result.to_csv(os.path.join(report_dir, f'{h}M_metrics_summary.csv'))
    for model in ['DT', 'RF', 'XGB']:
        fig = plot_regimes(
            date_series=res_rolling_all['Date'],
            data_series=res_rolling_all[model],
            regime_series=res_rolling_all['Regime'],
            area_name='Crash',
            data_name=model,
            x_axis_title='Date',
            y_axis_title='Crash Probability',
            plot_title=f'Probability of Crash Predicted by {model}',
            log_scale=False
        )
        fig.write_image(os.path.join(report_dir, f'{model}_crash_probs.png'))
        fig = plot_confusion_matrix(
            y_true=res_rolling_all['Regime'].values,
            y_pred=(res_rolling_all[model] > 0.5).astype(float).values,
            class_names=['Normal', 'Crash'],
            title=f'{model} Confusion Matrix'
        )
        fig.write_image(os.path.join(report_dir, f'{model}_confusion_matrix.png'))


def analyze_feature_importance(df, feature_cols, target_col, split_date, xgb_params, roll_window=150):
    oos_start_index = df[df['Date'] == split_date].index[0]
    df_pred = df.iloc[oos_start_index - roll_window:, :]
    X = df_pred[feature_cols]
    y = df_pred[target_col]
    date_range = df_pred['Date']
    feature_importance_list = []
    for i in np.arange(0, len(df_pred) - roll_window):
        xgb_model = xgb.XGBClassifier(**xgb_params)
        X_fit = X.iloc[i: i + roll_window, :]
        y_fit = y.iloc[i: i + roll_window]
        xgb_model = xgb_model.fit(X_fit, y_fit)
        importance_scores = xgb_model.feature_importances_
        true_label = y.iloc[i + roll_window]
        xgb_prob = xgb_model.predict_proba(X.iloc[i + roll_window: i + roll_window + 1, :])[:, 1][0]
        feature_importance_dict = {'Date': date_range.iloc[i + roll_window],
                                   'Regime': true_label,
                                   'Pred_Prob': xgb_prob}
        for j, feature in enumerate(feature_cols):
            feature_importance_dict[feature] = importance_scores[j]
        feature_importance_list.append(feature_importance_dict)
    feature_importance_df = pd.DataFrame(feature_importance_list)
    avg_feat_impo = pd.DataFrame(feature_importance_df[feature_cols].mean()).reset_index()
    avg_feat_impo.columns = ['Feature', 'AvgImportance']
    avg_feat_impo.sort_values('AvgImportance', ascending=False, inplace=True)
    feat_spec_df = pd.read_csv('./data/raw_data/fredmd_feat_spec.csv', index_col='ID')[['Feature', 'Group']]
    remove_texts = ['1M Lag', '3M Lag', '6M Lag', '9M Lag', '12M Lag']
    avg_feat_impo['CleanFeature'] = avg_feat_impo['Feature'].str.replace('|'.join(remove_texts), '', regex=True).str.strip()
    avg_feat_impo = avg_feat_impo.merge(feat_spec_df, left_on='CleanFeature', right_on='Feature', how='left').dropna()
    avg_feat_impo = avg_feat_impo[['Feature_x', 'Group', 'AvgImportance']]
    avg_feat_impo.columns = ['Feature', 'Group', 'Importance']
    avg_feat_impo.set_index('Feature')
    avg_group_impo = avg_feat_impo.groupby('Group')['Importance'].sum().reset_index()
    avg_group_impo.sort_values(by='Importance', ascending=False, inplace=True)
    avg_group_impo.set_index('Group')
    return avg_feat_impo


if __name__ == "__main__":
    split_date = '1973-01-01'
    target_col = 'MktRegime'
    h = 1  # horizon
    df, df_train, df_oos, feature_cols = load_market_regimes_data(target_col=target_col, split_date=split_date, horizon=h)
    plot_class_imbalance(df_oos, target_col)
    best_params_dict = cross_validate_models(df_train, feature_cols, target_col)
    res_rolling_all = rolling_oos_predictions(df, best_params_dict, feature_cols, target_col, split_date)
    res_rolling_all.set_index('Date').to_csv(f'./data/predictions/{h}M_mkt_preds.csv')
    metrics_result = plot_model_results(res_rolling_all)
    save_model_evaluation_reports(res_rolling_all, metrics_result, h=h, report_dir='./data/report/market')
    xgb_params = best_params_dict[('XGB', xgb.XGBClassifier)]
    feat_imp_df = analyze_feature_importance(df, feature_cols, target_col, split_date, xgb_params)

    report_dir = './data/report/market'
    feat_imp_df.to_csv(os.path.join(report_dir, f'{h}M_feature_importance_market.csv'), index=False)

    avg_feat_impo = feat_imp_df

    fig_feat = plot_feature_importance(feat_impo_df=avg_feat_impo,
                                       group_impo_df=avg_feat_impo.groupby('Group', as_index=False).sum(),
                                       plot_title='Average Feature Importance (Market)',
                                       bar_plot_title='Top Features',
                                       pie_plot_title='Importance by Group',
                                       start_feat_idx=1,
                                       max_feat_no=30)
    fig_feat.write_image(os.path.join(report_dir, f'{h}M_feature_importance_market.png'))