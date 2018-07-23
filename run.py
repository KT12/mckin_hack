import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.neighbors import KNeighborsRegressor

# https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon-4/

np.random.seed(3)

def effort(incentives):
    return 10*(1-np.exp(np.negative(incentives)/400))

def percent(effort):
    return 20*(1-np.exp(np.negative(effort)/5))

def p_e(incentives):
    return 20*(1-np.exp(np.negative(10*(1-np.exp(np.negative(incentives)/400)))/5))

def first_derive(incentives):
    return (np.exp(2*np.exp(np.negative(incentives)/400) - incentives/400 -2))/10

def feat_eng_train(fn='train_ZoGVYWq.csv'):
    # Load X set
    X = pd.read_csv(fn)
    # Rename vars
    X.rename(columns={
        'Income' : 'income',
        'age_in_days' : 'age',
        'perc_premium_paid_by_cash_credit' : 'trad_payment',
        'Count_3-6_months_late' : 'late_3_6m',
        'Count_6-12_months_late' : 'late_6_12m',
        'Count_more_than_12_months_late' : 'late_12m',
        'application_underwriting_score' : 'app_score',
        'no_of_premiums_paid' : 'count_premiums_paid'
        },
        inplace=True)
    # Target variable
    y = X['renewal'].copy()
    X.drop(labels=['id','renewal'], axis=1, inplace=True)
    
    # Rescale non_zero columns
    X['age'] = X['age'] / 365
    X['income'] = np.log(X['income'])

    # binarize urban
    X['urban'] = X['residence_area_type'].isin(['Urban'])
    X.drop(labels=['residence_area_type'], axis=1, inplace=True)
    # encode sourcing - remember to drop E when running prediction (multicollinearity)
    X['source_a'] = X['sourcing_channel'].isin(['A'])
    X['source_b'] = X['sourcing_channel'].isin(['B'])
    X['source_c'] = X['sourcing_channel'].isin(['C'])
    X['source_d'] = X['sourcing_channel'].isin(['D'])
    X['afford'] = X['income'] / (12 * X['premium'] )
    X.drop(labels=['sourcing_channel'], axis=1, inplace=True) 
    # Don't normalize binary variables
    bi_var = ['urban','source_a', 'source_b', 'source_c', 'source_d']
    nan_var = ['late_3_6m', 'late_6_12m', 'late_12m', 'app_score', 'premium_differential']
    scale_var = ['trad_payment', 'premium', 'count_premiums_paid', 'age']
    # Rescale scale_var
    rescale_dict = dict()
    rescale_dict['trad_payment_med'] = X['trad_payment'].median()
    rescale_dict['trad_payment_std'] = X['trad_payment'].std()
    rescale_dict['premium_med'] = X['premium'].median()
    rescale_dict['premium_std'] = X['premium'].std()
    rescale_dict['afford_med'] = X['afford'].median()
    rescale_dict['afford_std'] = X['afford'].std()
    rescale_dict['count_premiums_paid_med'] = X['count_premiums_paid'].median()
    rescale_dict['count_premiums_paid_std'] = X['count_premiums_paid'].std()
    rescale_dict['age_med'] = X['age'].median()
    rescale_dict['age_std'] = X['age'].std()
    for col in scale_var:
        X[col] = (X[col] - X[col].median())/X[col].std()
    
    # Fill missing data
    # Predict missing data later with more time!

    # X data has missing values in late, app_score
    y_val = X[~X['late_3_6m'].isnull()]['late_3_6m'].copy()
    x_val = X[~X['late_3_6m'].isnull()].drop(['late_3_6m', 'late_6_12m', 'late_12m', 'app_score'], axis=1).copy()
    x_full = X[~X['late_3_6m'].isnull()].copy()
    var_fit_3 = KNeighborsRegressor(n_neighbors=7, weights='distance')
    var_fit_3.fit(x_val, y_val)
    x_null = X[X['late_3_6m'].isnull()].copy()
    pred_late_3 = var_fit_3.predict(X[X['late_3_6m'].isnull()].drop(['late_3_6m', 'late_6_12m', 'late_12m', 'app_score'], axis=1))
    x_null['late_3_6m'] = pred_late_3
    X = pd.concat([x_full, x_null])

    # Predict 6-12
    y_val = X[~X['late_6_12m'].isnull()]['late_6_12m'].copy()
    x_val = X[~X['late_6_12m'].isnull()].drop(['late_6_12m', 'late_12m', 'app_score'], axis=1).copy()
    x_full = X[~X['late_6_12m'].isnull()].copy()
    var_fit_6 = KNeighborsRegressor(n_neighbors=7, weights='distance')
    var_fit_6.fit(x_val, y_val)
    x_null = X[X['late_6_12m'].isnull()].copy()
    pred_late_6 = var_fit_6.predict(X[X['late_6_12m'].isnull()].drop(['late_6_12m', 'late_12m', 'app_score'], axis=1))
    x_null['late_6_12m'] = pred_late_6
    X = pd.concat([x_full, x_null])

    # Predict 12
    y_val = X[~X['late_12m'].isnull()]['late_12m'].copy()
    x_val = X[~X['late_12m'].isnull()].drop(['late_12m', 'app_score'], axis=1).copy()
    x_full = X[~X['late_12m'].isnull()].copy()
    var_fit_12 = KNeighborsRegressor(n_neighbors=7, weights='distance')
    var_fit_12.fit(x_val, y_val)
    x_null = X[X['late_12m'].isnull()].copy()
    pred_late_12 = var_fit_12.predict(X[X['late_12m'].isnull()].drop(['late_12m', 'app_score'], axis=1))
    x_null['late_12m'] = pred_late_12
    X = pd.concat([x_full, x_null])


    # Predict App score    
    y_val = X[~X['app_score'].isnull()]['app_score'].copy()
    x_val = X[~X['app_score'].isnull()].drop(['app_score'], axis=1).copy()
    x_full = X[~X['app_score'].isnull()].copy()
    var_fit_app = KNeighborsRegressor(n_neighbors=7, weights='distance')
    var_fit_app.fit(x_val, y_val)
    x_null = X[X['app_score'].isnull()].copy()
    pred_late_app = var_fit_app.predict(X[X['app_score'].isnull()].drop(['app_score'], axis=1))
    x_null['app_score'] = pred_late_app
    X = pd.concat([x_full, x_null])


    X['inverse'] =  X['late_6_12m'] > (1/X['late_12m'])
    # Return to original form and compute
    X['premium_differential'] = \
        ((X['count_premiums_paid'] * rescale_dict['count_premiums_paid_std'])/rescale_dict['count_premiums_paid_med'])/( ((X['age'] * rescale_dict['age_std'])/rescale_dict['age_med'])-20) - X['late_3_6m'] - (1/2)* X['late_6_12m'] - (1/4) * X['late_12m']

    # Rescale missing value cols
    rescale_dict['late_3_6m_med'] = X['late_3_6m'].median()
    rescale_dict['late_3_6m_std'] = X['late_3_6m'].std()
    rescale_dict['late_6_12m_med'] = X['late_6_12m'].median()
    rescale_dict['late_6_12m_std'] = X['late_6_12m'].std()
    rescale_dict['late_12m_med'] = X['late_12m'].median()
    rescale_dict['late_12m_std'] = X['late_12m'].std()
    rescale_dict['app_score_med'] = X['app_score'].median()
    rescale_dict['app_score_std'] = X['app_score'].std()
    rescale_dict['premium_differential_med'] = X['premium_differential'].median()
    rescale_dict['premium_differential_std'] = X['premium_differential'].std()


    for col in nan_var:
        X[col] = (X[col] - X[col].median())/X[col].std()

    print(X.columns)
    return X, y, rescale_dict, var_fit_3, var_fit_6, var_fit_12, var_fit_app


def feat_eng_test(fn, rescale_dict, var_fit_3, var_fit_6, var_fit_12, var_fit_app):
    # Load X set
    X = pd.read_csv(fn)
    # Rename vars
    X.rename(columns={
        'Income' : 'income',
        'age_in_days' : 'age',
        'perc_premium_paid_by_cash_credit' : 'trad_payment',
        'Count_3-6_months_late' : 'late_3_6m',
        'Count_6-12_months_late' : 'late_6_12m',
        'Count_more_than_12_months_late' : 'late_12m',
        'application_underwriting_score' : 'app_score',
        'no_of_premiums_paid' : 'count_premiums_paid'
        },
        inplace=True)
    # Target variable
    y  = X['id']
    X.drop(labels=['id'], axis=1, inplace=True) 
    # Rescale non_zero columns
    X['age'] = X['age'] / 365
    X['income'] = np.log(X['income'])

    # binarize urban
    X['urban'] = X['residence_area_type'].isin(['Urban'])
    X.drop(labels=['residence_area_type'], axis=1, inplace=True)
    # encode sourcing - remember to drop E when running prediction (multicollinearity)
    X['source_a'] = X['sourcing_channel'].isin(['A'])
    X['source_b'] = X['sourcing_channel'].isin(['B'])
    X['source_c'] = X['sourcing_channel'].isin(['C'])
    X['source_d'] = X['sourcing_channel'].isin(['D'])
    X['afford'] = X['income'] / (12 * X['premium'] )
    X.drop(labels=['sourcing_channel'], axis=1, inplace=True) 
    # Don't normalize binary variables
    bi_var = ['urban','source_a', 'source_b', 'source_c', 'source_d']
    nan_var = ['late_3_6m', 'late_6_12m', 'late_12m', 'app_score', 'premium_differential']
    scale_var = ['trad_payment', 'premium', 'count_premiums_paid', 'age']

    for col in scale_var:
        X[col] = (X[col] - X[col].median())/X[col].std()

    # X data has missing values in late, app_score
    # Fill 3
    x_full = X[~X['late_3_6m'].isnull()].copy()
    x_null = X[X['late_3_6m'].isnull()].copy()
    pred_late_3 = var_fit_3.predict(X[X['late_3_6m'].isnull()].drop(['late_3_6m', 'late_6_12m', 'late_12m', 'app_score'], axis=1))
    x_null['late_3_6m'] = pred_late_3
    X = pd.concat([x_full, x_null])

    # Predict 6-12
    x_full = X[~X['late_6_12m'].isnull()].copy()
    x_null = X[X['late_6_12m'].isnull()].copy()
    pred_late_6 = var_fit_6.predict(X[X['late_6_12m'].isnull()].drop(['late_6_12m', 'late_12m', 'app_score'], axis=1))
    x_null['late_6_12m'] = pred_late_6
    X = pd.concat([x_full, x_null])

    # Predict 12
    x_full = X[~X['late_12m'].isnull()].copy()
    x_null = X[X['late_12m'].isnull()].copy()
    pred_late_12 = var_fit_12.predict(X[X['late_12m'].isnull()].drop(['late_12m', 'app_score'], axis=1))
    x_null['late_12m'] = pred_late_12
    X = pd.concat([x_full, x_null])

    # Predict App score    
    x_full = X[~X['app_score'].isnull()].copy()
    x_null = X[X['app_score'].isnull()].copy()
    pred_late_app = var_fit_app.predict(X[X['app_score'].isnull()].drop(['app_score'], axis=1))
    x_null['app_score'] = pred_late_app
    X = pd.concat([x_full, x_null])

    X['inverse'] =  X['late_6_12m'] > (1/X['late_12m'])
    X['premium_differential'] = \
        ((X['count_premiums_paid'] * rescale_dict['count_premiums_paid_std'])/rescale_dict['count_premiums_paid_med'])/( ((X['age'] * rescale_dict['age_std'])/rescale_dict['age_med'])-20) - X['late_3_6m'] - (1/2)* X['late_6_12m'] - (1/4) * X['late_12m']
    
    # Now use rescale_dict to rescale all variables
    X['trad_payment'] = (X['trad_payment'] - rescale_dict['trad_payment_med']) / rescale_dict['trad_payment_std']
    X['premium']  = (X['premium'] - rescale_dict['premium_med']) / rescale_dict['premium_std']
    X['late_3_6m'] = (X['late_3_6m'] - rescale_dict['late_3_6m_med']) / rescale_dict['late_3_6m_std']
    X['late_6_12m'] = (X['late_6_12m'] - rescale_dict['late_6_12m_med']) / rescale_dict['late_3_6m_std']
    X['late_12m'] = (X['late_12m'] - rescale_dict['late_12m_med']) / rescale_dict['late_12m_std']
    X['app_score'] = (X['app_score'] - rescale_dict['app_score_med']) / rescale_dict['app_score_std']
    X['premium_differential'] = (X['premium_differential'] - rescale_dict['premium_differential_med']) / rescale_dict['premium_differential_std']
    X['afford'] = (X['afford'] - rescale_dict['afford_med']) / rescale_dict['afford_std']
    X['count_premiums_paid'] = (X['count_premiums_paid'] - rescale_dict['count_premiums_paid_med']) / rescale_dict['count_premiums_paid_std']
    print(X.columns)
    return X, y


def rf(X, y):
    # Define classifier='test_66516Ee.csv'
    clf = RandomForestClassifier(criterion='gini', n_jobs=-1)
    param_grid = {
    'n_estimators': [n*2 for n in range(5,10)],
    'max_depth': [n for n in range(4,8)],
    'min_samples_split':[512, 256, 128],
    'min_samples_leaf':[81, 64, 32]}
    
    
    clf_grid = GridSearchCV(estimator=clf, cv=4, param_grid=param_grid, scoring='roc_auc')
    pipeline = make_pipeline(clf_grid)
    pipeline.fit(X=X, y=y)
    best_clf = clf_grid.best_estimator_
    return best_clf

def main():
    X, y, rescale_dict, var_fit_3, var_fit_6, var_fit_12, var_fit_app = feat_eng_train()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    best_rf_clf =  rf(X, y)
    # Compute ROC_AUC
    roc = roc_auc_score(y, np.array([x[1] for x in best_rf_clf.predict_proba(X)]))
    print('ROCAUC Score: {}'.format(roc))
    X_data, y_ids = feat_eng_test('test_66516Ee.csv', rescale_dict, var_fit_3, var_fit_6, var_fit_12, var_fit_app)
    y_pred = np.array([x[1] for x in best_rf_clf.predict_proba(X_data)])
    ids = pd.read_csv('test_66516Ee.csv')['id']
    y_submit = pd.DataFrame(data={'id':ids, 'renewal':y_pred})
    y_submit.to_csv('halfway.csv', index=False)
    # Append probs to X_data
    X_data['prob'] = y_pred
    X_data['max_prob_add'] = 1 - X_data['prob']
    X_part_2  = X_data[['premium', 'prob','max_prob_add']].copy()
    # Rescale premium
    X_part_2['premium'] = (X_part_2['premium'] * rescale_dict['premium_std']) + rescale_dict['premium_med']

    # Create incentives df
    incent = np.array(range(0,20000))
    ppp = p_e(incent)
    effort_df = pd.DataFrame({'incent':incent, 'ppp':ppp})
    # Find appropriate level of incentives for each customer, based on expected premium
    incentives = []
    for k in X_part_2.itertuples():
        calcs_cost = effort_df[effort_df['incent']< k.premium/3].copy()
        calcs_cost_2 = effort_df[effort_df['ppp']/100< k.max_prob_add].copy()
        calc_df = min([calcs_cost, calcs_cost_2], key=len)
        calc_df['expected_revenue'] = k.premium * (calc_df['ppp'] + k.prob) - calc_df['incent']
        try:
            incentives.append(calc_df.loc[calc_df['expected_revenue'].idxmax()]['incent'])
        except ValueError:
            incentives.append(0)
    X_part_2['incent_guess'] = incentives
    y_submit['incentives'] = incentives
    y_submit.to_csv('submit_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.csv', index=False)# Compute combined delinquent time
if __name__ == '__main__':
    main()