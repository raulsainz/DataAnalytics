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



    

