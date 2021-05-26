# =============================================================================
# Created By  : Raul Sainz
# Created Date: 2021-03-21
# =============================================================================
# Imports
# =============================================================================
from termcolor import colored   #Function to print console message with colors
import datetime                 #Library for getting tim
import requests                 #Library allows to send send HTTP requests
from urllib import parse        #Library to make URL request to Wikipedia API
from googletrans import Translator, constants
from os import path
import requests
import pandas as pd
#Libraries for model evaluation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
#Define constant enviroment variables
DATASET_PATH = 'datasets/'
GITHUB_URL = 'https://raw.githubusercontent.com/raulsainz/DataAnalytics/main/datasets/'

# =============================================================================
# Function logMessage
# Gets and prints a message with a color depending on the level
# @msg str: Message to be printed
# @level int: Type of message 
#                            0 - default
#                            1 - OK (green output)
#                            2 - Error (red )
#                            2 - Warning (red )
#                            2 - Notice (red )                           
# @returns: Print console message
# =============================================================================

def logMessage(msg = 'Empty msg',level = 0,addTime = True): 
    if addTime:
        date = '[' + datetime.datetime.now().strftime("%m/%d/%Y-%H:%M:%S") + ']'
        date = colored(str(date)+': ', 'white', attrs=['reverse', 'blink'])
    else:
        date='[-]'
    if level == 0: # Default informative output
        print(date+msg)
    if level == 1: # OK output
        print(date+colored('OK: '+ msg, 'green', attrs=['reverse', 'blink']))
    if level == 2: # Error output
        print(date+colored('ERROR: '+ msg, 'red', attrs=['reverse', 'blink']))
    if level == 3: # Warning output
        print(date+colored('WARNING: '+ msg, 'yellow', attrs=['reverse', 'blink']))
    if level == 4: # Important output
        print(date+colored('NOTICE: '+ msg, 'blue', attrs=['reverse', 'blink']))
# =============================================================================
# Function confirmAction
# Ask for confirmation on a given question
# @msg str: Question to be asked for confirmation
# @returns: True if y, False everything else
# =============================================================================

def confirmAction(msg):
    reply = str(input(msg+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        logMessage('Action confirmed',1)
        return False
    else:
        logMessage('Action rejected',3)
        return True



def loadDataSet(filename,sample_size=1,describe=True,git = False):
    try: 
        if path.exists(DATASET_PATH+filename) & (not git) : #Check if file exist locally
            df = pd.read_csv(DATASET_PATH+filename)
        else:
            logMessage('File {} not found on local folder loading from github...',3)
            response = requests.get(GITHUB_URL+filename)
            # If the response was successful, no Exception will be raised
            response.raise_for_status()
            print(response)
            df = pd.read_csv(GITHUB_URL+filename)
        df_rows = df.shape[0]
        df_cols = df.shape[1]
        logMessage('succesfuly loaded data frame -- rows: {} - columns: {}.'.format(df_rows,df_cols),1)
        
        if sample_size<1:   
            df = df.sample(frac=sample_size, replace=True, random_state=1)
            df_rows = df.shape[0]
            df_cols = df.shape[1]
            logMessage('sampling data frame at {}% -- rows: {} - columns: {}.'.format(sample_size*100,df_rows,df_cols))
        null_values = df.isnull().values.any()
        if null_values:
            logMessage('Null Values Found!!',3)
            for col in df.columns:
                nul_num = df[col].isnull().sum()
                if nul_num>0:
                    por_num = (nul_num/df_rows)*100
                    logMessage('Column {} has {} ({:.2f}%) Null values.'.format(col,nul_num,por_num))
        else:
            logMessage('Data Frame has 0 null values {}.'.format(null_values),1)

        return df
    except Exception as e:
        logMessage(str(e),2)

class MyCustomError(Exception):
    pass
def getAirportInfo(code):
    url = "https://airport-info.p.rapidapi.com/airport"
    headers = {
                'x-rapidapi-key': "3c8298d3abmsh47bededcce49e9ep16b1bbjsn8b382dd0e6c9",
                'x-rapidapi-host': "airport-info.p.rapidapi.com"
            }
    params = {"iata":code}
    response = requests.request("GET", url, headers=headers , params=params)
    try:     
        if response.status_code ==200:           
            data = response.json()
            if 'ageYears' in data:
                age = data['ageYears']
            else:
                return [tail,np.NaN]
            if type(age) == int or type(age) == float:
                print('Aircraft:'+tail+' Age:'+ str(data['ageYears']))
                return [tail,age]
            else:
                print('Aircraft:'+tail+' Age: NA')
                return [tail,np.NaN]
        else:
            raise MyCustomError("bad response: "+str(response.status_code))
            return [tail,np.NaN]
        #print('Aircraft:'+tail+' Age:'+data['ageYears']) 
    except Exception as e:
        return "NO DATA"
    return    


def run_model2(model, X_train, y_train, X_test, y_test, verbose=True, desc = 'No Name',labels = ['True','False']):
    global model_results
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    model_acc = metrics.accuracy_score(y_test, y_pred )  # Model Accuracy, how often is the classifier correct?
    model_kappa = cohen_kappa_score(y_test, y_pred ) 
    class_report = classification_report(y_test,y_pred,digits=2,output_dict=True)
    cross_val_score_mean = mean(cross_val_score(model,X_train,y_train,cv=cv_n_split))



    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Model Results for: {}'.format(desc))
    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all', display_labels = labels, values_format = '.2%',colorbar = False,ax=ax[0]) 
    plot_roc_curve(model, X_test, y_test, ax = ax[1])

    print("============== Results for {} =================".format(desc))
    print("Accuracy              : {:.2%}".format(model_acc))
    print("ROC_AUC               : {:.2%}".format(roc_auc))
    print("KAPPA                 : {:.2%}".format(model_kappa))
    print("MacroAVG-Precision    : {:.2%}".format(class_report['macro avg']['precision']))
    print("MacroAVG-Recall       : {:.2%}".format(class_report['macro avg']['recall']))
    print("macroAVG-F1 Score     : {:.2%}".format(class_report['macro avg']['f1-score']))
    print("Avg CrossValid Score  : {:.2%} ({} Folds)".format(cross_val_score_mean,cv_n_split))

    fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred)
    model_results[len(model_results)] = {
        'model'                 :type(model),
        'description'           :desc,
        'model_acc'             :model_acc,
        'roc_auc'               :roc_auc,
        'cross_val_score_mean'  :cross_val_score_mean,
        'Precision'             :class_report['macro avg']['precision'],
        'Recall'                :class_report['macro avg']['recall'],
        'f1_score'              :class_report['macro avg']['f1-score'],
        'fpr'                   :fpr,
        'tpr'                   :tpr
    }
    

