import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


#allows to show the dataframes better
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


data = pd.read_csv('cbb.csv')

#randomly shuffle data
data = data.sample(frac=1)




#create variable for unranked teams
rs = pd.isna(data['SEED'])
unranked = data[rs]
#print(unranked)

#create variable for ranked teams
rs = pd.notna(data['SEED'])
ranked = data[rs]
#print(ranked)


#reset indexes
unranked = unranked.reset_index(drop=True)
#print(unranked)
ranked = ranked.reset_index(drop=True)
#print(ranked)


pre = preprocessing.LabelEncoder()


data['CONF'] = pre.fit_transform(data['CONF'])
data['TEAM'] = pre.fit_transform(data['TEAM'])
data = data.drop(['POSTSEASON', 'SEED'], axis=1)

yr2015 = data[data['YEAR']==2015]
yr2016 = data[data['YEAR']==2016]
yr2017 = data[data['YEAR']==2017]
yr2018 = data[data['YEAR']==2018]
yr2019 = data[data['YEAR']==2019]
yr2020 = data[data['YEAR']==2020]






unranked = unranked.drop(['POSTSEASON','SEED'], axis= 1)
unranked['CONF'] = pre.fit_transform(unranked['CONF'])
unranked['TEAM'] = pre.fit_transform(unranked['TEAM'])
#unranked= unranked[1:]
#print(unranked)

ranked = ranked.drop(['TEAM'], axis= 1)
ranked['CONF'] = pre.fit_transform(ranked['CONF'])
ranked['POSTSEASON'] = pre.fit_transform(ranked['POSTSEASON'])



def results(y_test,y_pred):
    for i in range(0,6):
        z = pd.DataFrame()
        z['Actual'] = y_test
        z['Predicted'] = y_pred
        z['Difference'] = abs(y_test - y_pred)
        diff = z['Difference']
        print('Accuracy within ' + str(i) + ' games is ' + str(round(diff[diff <= i].count() / z.Difference.count() * 100,2)) + '%')
        best_result = round(diff[diff <= i].count() / z.Difference.count() * 100,2)


def run_tests(dataset_name,dataset):

    X = dataset.drop(['W'], axis=1)
    y= dataset.W

    gnb = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 70% training and 30% test

    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    #print( dataset_name + " Naives Bayes Accuracy:", accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    sn.heatmap(confusion_matrix(y_test, y_pred), annot=True)  # font size
    plt.xlabel('true label')
    plt.ylabel('predicted label');
#    plt.show()

    #-----------------------------------------------------------------------------------

    # Random Forsest

    rfc_b = RFC()
    rfc_b.fit(X_train, y_train)
    y_pred = rfc_b.predict(X_train)
    #print(dataset_name + ' Random Forest Train accuracy score:', accuracy_score(y_train, y_pred))
    #print(dataset_name + ' Random Forest Test accuracy score:', accuracy_score(y_test, rfc_b.predict(X_test)))
    print(accuracy_score(y_test, rfc_b.predict(X_test)))

    #-----------------------------------------------------------------------------------


    # Ridge Regression

    ridge = Ridge()
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_train)
    y_pred = np.asarray([np.argmax(line) for line in y_pred])
    yp_test = ridge.predict(X_test)
    test_preds = np.asarray([np.argmax(line) for line in yp_test])
    #print(dataset_name + ' Ride Regression Train accuracy score:', accuracy_score(y_train, y_pred))
    #print(dataset_name + ' Ride Regression Test accuracy score:', accuracy_score(y_test, test_preds))
    #print(accuracy_score(y_test, test_preds))
    print(accuracy_score(y_test, test_preds))




    #-----------------------------------------------------------------------------------


    # K-Nearest Neighbors

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    #print(dataset_name + ' K-NN Train accuracy score:', accuracy_score(y_train, y_pred))
    #print(dataset_name + ' K-NN Test accuracy score:', accuracy_score(y_test, knn.predict(X_test)))
    print(accuracy_score(y_test, knn.predict(X_test)))

    #-----------------------------------------------------------------------------------

    abc = AdaBoostClassifier(n_estimators=500)
    # Train Adaboost Classifer
    model = abc.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    #print(dataset_name + ' AdaBoost Test accuracy score:', metrics.accuracy_score(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))


    #-----------------------------------------------------------------------------------

    gbc = GradientBoostingClassifier(n_estimators=500)
    # Train Adaboost Classifer
    model = gbc.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    #results(y_test,y_pred)

    # Model Accuracy, how often is the classifier correct?
    #print(dataset_name + ' GradientBoosting Test accuracy score:', metrics.accuracy_score(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))

#-----------------------------------------------------------------------------------

    gbr = GradientBoostingRegressor(n_estimators=500)
    # Train Adaboost Classifer
    model = gbr.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    #results(y_test, y_pred)
    # Model Accuracy, how often is the classifier correct?
    #print(dataset_name + ' GradientRegression Test accuracy score:', metrics.accuracy_score(y_test, y_pred.round()))
    print(metrics.accuracy_score(y_test, y_pred.round()))






run_tests('Unranked', unranked)
run_tests('Full Dataset', data)
run_tests('Ranked', ranked)
run_tests('2015', yr2015)
run_tests('2016', yr2016)
run_tests('2017', yr2017)
run_tests('2018', yr2018)
run_tests('2019', yr2019)




data_specific_cols = data[['W','WAB','BARTHAG','ADJOE','G','ADJDE','EFG_O','EFG_D','2P_O','2P_D']]
unranked_specific_cols = unranked[['W','WAB','BARTHAG','ADJOE','G','ADJDE','EFG_O','EFG_D','2P_O','2P_D']]
ranked_specific_cols = ranked[['W','WAB','BARTHAG','ADJOE','G','ADJDE','EFG_O','EFG_D','2P_O','2P_D']]
yr2015_specific_cols = yr2015[['W','WAB','BARTHAG','ADJOE','G','ADJDE','EFG_O','EFG_D','2P_O','2P_D']]
yr2016_specific_cols = yr2016[['W','WAB','BARTHAG','ADJOE','G','ADJDE','EFG_O','EFG_D','2P_O','2P_D']]
yr2017_specific_cols = yr2017[['W','WAB','BARTHAG','ADJOE','G','ADJDE','EFG_O','EFG_D','2P_O','2P_D']]
yr2018_specific_cols = yr2018[['W','WAB','BARTHAG','ADJOE','G','ADJDE','EFG_O','EFG_D','2P_O','2P_D']]
yr2019_specific_cols = yr2019[['W','WAB','BARTHAG','ADJOE','G','ADJDE','EFG_O','EFG_D','2P_O','2P_D']]
yr2020_specific_cols = yr2020[['W','WAB','BARTHAG','ADJOE','G','ADJDE','EFG_O','EFG_D','2P_O','2P_D']]


run_tests('Unranked', unranked_specific_cols)
run_tests('Full Dataset', data_specific_cols)
run_tests('Ranked', ranked_specific_cols)
run_tests('2015', yr2015_specific_cols)
run_tests('2016', yr2016_specific_cols)
run_tests('2017', yr2017_specific_cols)
run_tests('2018', yr2018_specific_cols)
run_tests('2019', yr2019_specific_cols)





