

#Libraries for model selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
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
#Other Libraries
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd


def run_classification_model(model, X, y, verbose = False, desc = 'No Name', labels = ['True', 'False'],cv_n_split = 5,random_state=42,test_train_split=0.25,suffle=True):
    bin_palette = sns.color_palette(["#0081a7","#f07167"])
    X_train,X_test, y_train,y_test = train_test_split(X , y , random_state = random_state, shuffle = suffle, test_size = test_train_split)

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    cross_val = cross_val_score(model,X_train,y_train,cv=cv_n_split)
    model_acc = metrics.accuracy_score(y_test, y_pred )  # Model Accuracy, how often is the classifier correct?
    model_kappa = cohen_kappa_score(y_test, y_pred ) 
    class_report = classification_report(y_test,y_pred,digits=2,output_dict=True)
    cross_val_score_mean = mean(cross_val)
    fig, ax = plt.subplots(ncols=2,nrows=2,figsize=(15,10))
    fig.suptitle('Model Results for: {}'.format(desc))
    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all', display_labels = labels, values_format = '.2%',colorbar = True,ax=ax[0,0]) 
    plot_roc_curve(model, X_test, y_test, ax = ax[0,1])
    sns.boxplot(y=cross_val,ax=ax[1,0],palette=bin_palette,width=0.2)
    sns.swarmplot(y = cross_val, color=bin_palette[1],ax=ax[1,0],size=10)
    ax[1,0].set(title="",xlabel='Cross Validation Results')
    ax[1,1].set(title="",xlabel='Model Scores')
    ax[0,1].set(title="ROC Curve Plot",xlabel='True Positive Rate',ylabel='False Positive Rate')
    ax[0,0].set(title="Confusion Matrix")
    
    
    toplot= {
        'model_acc'             :model_acc,
        'roc_auc'               :roc_auc,
        'KAPPA'                 :model_kappa,
        'crossval_mean'  :cross_val_score_mean,
        'Precision'             :class_report['macro avg']['precision'],
        'Recall'                :class_report['macro avg']['recall'],
        'f1_score'              :class_report['macro avg']['f1-score']
        }
    #ax[1,1] = plt.bar(toplot.keys(), toplot.values())
    ax[1,1] = plt.step(toplot.values(),toplot.keys(), label='pre (default)',color='#0081a7')
    ax[1,1] = plt.plot(toplot.values(),toplot.keys(), 'o--', alpha=0.7,color='#f07167')
    #plt.xlim(0.8,1)
    #ax.set(xlim = (50000,250000))
    
    
    if verbose:
        print("============== Results for {} =================".format(desc))
        print("Accuracy              : {:.2%}".format(model_acc))
        print("ROC_AUC               : {:.2%}".format(roc_auc))
        print("KAPPA                 : {:.2%}".format(model_kappa))
        print("MacroAVG-Precision    : {:.2%}".format(class_report['macro avg']['precision']))
        print("MacroAVG-Recall       : {:.2%}".format(class_report['macro avg']['recall']))
        print("macroAVG-F1 Score     : {:.2%}".format(class_report['macro avg']['f1-score']))
        print("Avg CrossValid Score  : {:.2%} ({} Folds)".format(cross_val_score_mean,cv_n_split))
        # The coefficients
        data = {"variables": list(X_train.columns), "Coefficients": model.coef_[0]}
        df_coefficients = pd.DataFrame(data)
        #df_coefficients = pd.DataFrame(model.coef_, columns = X_train.columns)
        print('Intercept: ', model.intercept_)          
        print('Coefficients:')
        print(df_coefficients)



    fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred)
    results = {
        'model'                 :type(model),
        'description'           :desc,
        'model_acc'             :model_acc,
        'roc_auc'               :roc_auc,
        'kappa'                 :model_kappa,
        'crossval__mean'  :cross_val_score_mean,
        'Precision'             :class_report['macro avg']['precision'],
        'Recall'                :class_report['macro avg']['recall'],
        'f1_score'              :class_report['macro avg']['f1-score'],
        'fpr'                   :fpr,
        'tpr'                   :tpr
    }
    return results
    
def printClassificationResults(results):
    df_results = pd.DataFrame.from_dict(results)
    df_results = df_results.T
    
    heat = df_results[['description','model_acc','roc_auc','kappa','crossval__mean','Precision','Recall','f1_score']]
    heat = heat.set_index('description')
    heat = heat.sort_values(by='model_acc', ascending=False)
    for col in heat.columns:
        heat[col] = heat[col].astype(float)
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(15, 8)
    fig.suptitle('Model Comparisson')
    sns.heatmap(heat,annot=True, ax = ax[0],fmt ='.0%',cmap=sns.diverging_palette(20, 220, n=200) )
    #ax[0].xticks(rotation=45)
    ax[0].set(xlabel="")
    for i in results:
        model = results[i]
        label = "{}-AUC:{:.2%}".format(model['description'],model['model_acc'],model['roc_auc'])
        ax[1] = plt.plot(model['fpr'],model['tpr'],label=label,linewidth=5,linestyle='dashed')
        plt.legend(loc=0,fontsize=12,frameon=True,bbox_to_anchor = (.2, .35),prop = {'weight':'bold'} )
        #plt.legend(loc=0,fontsize=15,frameon=True,bbox_to_anchor = (.2, .35),prop = {'weight':'bold'} )
        #ax[1].set(title="ROC Curve Plot",xlabel='True Positive Rate',ylabel='False Positive Rate')
