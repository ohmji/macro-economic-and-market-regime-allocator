# General
import pandas as pd
import numpy as np
import itertools
import os

# Visualization
import plotly.express as px
from src.data_understanding import plot_regimes

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Model Evaluation
from src.model_evaluation import plot_confusion_matrix, evaluation_metrics, plot_feature_importance



def load_and_prepare_data(path='./data/datasets/econ_regs_ds.csv', target_col='EconRegime', horizon=1, split_date='1973-01-01'):
    df = pd.read_csv(path)
    feature_cols = df.columns.drop([target_col, 'Date'])
    df[feature_cols] = df[feature_cols].shift(+horizon)
    df.dropna(inplace=True)
    df_train = df[df['Date'] < split_date]
    df_oos = df[df['Date'] >= split_date]
    return df, df_train, df_oos, feature_cols


def perform_cross_validation(df_train, feature_cols, target_col):
    model_dict_cv = {
        ('DT', DecisionTreeClassifier): {'max_depth': [3, 5, 8, 10], 'splitter': ['best', 'random'], 'min_samples_split': [2, 3, 5]},
        ('RF', RandomForestClassifier): {'random_state': [42], 'max_depth': [3, 5, 8, 10], 'n_estimators': [100, 200, 400]},
        ('XGB', xgb.XGBClassifier): {'booster': ['gbtree'], 'max_depth': [3, 5, 8, 10], 'n_estimators': [100, 200, 400], 'random_state': [42], 'objective': ['binary:logistic']}
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
                X_train = df_train[feature_cols].iloc[train_index]
                y_train = df_train[target_col].iloc[train_index]
                X_test = df_train[feature_cols].iloc[test_index]
                y_test = df_train[target_col].iloc[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score.append((y_pred == y_test).mean())

            cv_scores.append(np.mean(score))

        best_params_dict[model_tuple] = all_grid[np.argmax(cv_scores)]
    return best_params_dict


def train_oos_predict(df, best_params_dict, feature_cols, target_col, split_date, horizon=1, roll_window=150):
    oos_start_index = df[df['Date'] == split_date].index[0]
    df_pred = df.iloc[oos_start_index - roll_window:, :]
    X, y = df_pred[feature_cols], df_pred[target_col]
    date_range = df_pred['Date']
    model_counter = 0

    for model_tuple, param in best_params_dict.items():
        y_prob, date, y_actual = np.array([]), np.array([], dtype='datetime64[s]'), np.array([])
        for i in range(len(df_pred) - roll_window):
            model = model_tuple[1](**param)
            model.fit(X.iloc[i:i + roll_window], y.iloc[i:i + roll_window])
            y_pred = model.predict_proba(X.iloc[i + roll_window:i + roll_window + 1])[:, 1]
            y_prob = np.hstack((y_prob, y_pred))
            date = np.hstack((date, date_range.iloc[i + roll_window:i + roll_window + 1].values))
            y_actual = np.hstack((y_actual, y.iloc[i + roll_window:i + roll_window + 1].values))

        if model_counter == 0:
            res_df = pd.DataFrame({'Date': date, 'Regime': y_actual, model_tuple[0]: y_prob})
            model_counter += 1
        else:
            res_df[model_tuple[0]] = y_prob

    return res_df


def evaluate_models(res_df):
    for model in ['DT', 'RF', 'XGB']:
        fig = plot_regimes(res_df['Date'], res_df[model], res_df['Regime'], area_name='Recession',
                           data_name=model, x_axis_title='Date', y_axis_title='Recession Probability',
                           plot_title=f'Probability of Recession Predicted by {model}', log_scale=False)
        fig.show()

        fig = plot_confusion_matrix(res_df['Regime'].values,
                                    (res_df[model] > 0.5).astype(int).values,
                                    class_names=['Normal', 'Recession'],
                                    title=f'{model} Confusion Matrix')
        fig.show()

    metrics_result, fig = evaluation_metrics(res_df, model_cols=['DT', 'RF', 'XGB'], y_true_col='Regime')
    fig.show()
    return metrics_result


def analyze_feature_importance(df, feature_cols, target_col, split_date, xgb_params, roll_window=150):
    oos_start_index = df[df['Date'] == split_date].index[0]
    df_pred = df.iloc[oos_start_index - roll_window:, :]
    X, y = df_pred[feature_cols], df_pred[target_col]
    date_range = df_pred['Date']
    feature_importance_list = []

    for i in range(len(df_pred) - roll_window):
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X.iloc[i:i + roll_window], y.iloc[i:i + roll_window])
        scores = model.feature_importances_
        prob = model.predict_proba(X.iloc[i + roll_window:i + roll_window + 1])[:, 1][0]
        record = {'Date': date_range.iloc[i + roll_window], 'Regime': y.iloc[i + roll_window], 'Pred_Prob': prob}
        for j, f in enumerate(feature_cols):
            record[f] = scores[j]
        feature_importance_list.append(record)

    return pd.DataFrame(feature_importance_list)


if __name__ == "__main__":
    split_date = '1973-01-01'
    target_col = 'EconRegime'
    h = 1

    df, df_train, df_oos, feature_cols = load_and_prepare_data(split_date=split_date, target_col=target_col, horizon=h)
    best_params = perform_cross_validation(df_train, feature_cols, target_col)
    results_df = train_oos_predict(df, best_params, feature_cols, target_col, split_date)
    results_df.set_index('Date').to_csv(f'./data/predictions/{h}M_econ_preds.csv')
    metrics_df = evaluate_models(results_df)
    xgb_params = best_params[('XGB', xgb.XGBClassifier)]
    feat_imp_df = analyze_feature_importance(df, feature_cols, target_col, split_date, xgb_params)


    # Create report directory
    report_dir = './report'
    os.makedirs(report_dir, exist_ok=True)

    # Save evaluation metrics CSV
    metrics_df.to_csv(os.path.join(report_dir, f'{h}M_metrics_summary.csv'))

    # Save confusion matrices and predicted probabilities plots
    for model in ['DT', 'RF', 'XGB']:
        fig = plot_regimes(results_df['Date'], results_df[model], results_df['Regime'], area_name='Recession',
                           data_name=model, x_axis_title='Date', y_axis_title='Recession Probability',
                           plot_title=f'Probability of Recession Predicted by {model}', log_scale=False)
        fig.write_image(os.path.join(report_dir, f'{model}_recession_probs.png'))

        fig = plot_confusion_matrix(results_df['Regime'].values,
                                    (results_df[model] > 0.5).astype(int).values,
                                    class_names=['Normal', 'Recession'],
                                    title=f'{model} Confusion Matrix')
        fig.write_image(os.path.join(report_dir, f'{model}_confusion_matrix.png'))

    feat_imp_df.to_csv(os.path.join(report_dir, f'{h}M_feature_importance_econ.csv'), index=False)

    from src.model_evaluation import plot_feature_importance
    avg_feat_impo = pd.DataFrame(feat_imp_df[feature_cols].mean()).reset_index()
    avg_feat_impo.columns = ['Feature', 'Importance']
    avg_feat_impo.sort_values('Importance', ascending=False, inplace=True)
    avg_feat_impo['Group'] = 'Unknown'  # Optional: map to actual groups if available

    fig_feat = plot_feature_importance(feat_impo_df=avg_feat_impo,
                                       group_impo_df=avg_feat_impo.groupby('Group', as_index=False).sum(),
                                       plot_title='Average Feature Importance (Economic)',
                                       bar_plot_title='Top Features',
                                       pie_plot_title='Importance by Group',
                                       start_feat_idx=1,
                                       max_feat_no=30)
    fig_feat.write_image(os.path.join(report_dir, f'{h}M_feature_importance_econ.png'))