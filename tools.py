import os as os
import graphlab as gl
import matplotlib.pyplot as plt


def pivot(data, row, col, item, agg=gl.aggregate.COUNT("id")):
    tab= data.groupby([col, row],
            {item:agg}).groupby([row],
            {"xyzzy":gl.aggregate.CONCAT(col, item)}).unpack('xyzzy')

    for col in tab.column_names():
        tab[col]=tab[col].fillna(0)

    col_names = tab.column_names()
    col_names.remove(row)
    col_dict = dict((col, col.replace("xyzzy.", '')) for col in col_names)
    
    tab.rename(col_dict)

    return tab


from re import match
def features_like(regex, data):
    return set([col for  col in data.column_names() if match(regex, col)])


def features_type(t, data):
    return set([col for  col in data.column_names() if data[col].dtype() == type(t)])

  
def data_dictionary(data, col_list=[]):
    
    if len(col_list) == 0:
        col_list = data.column_names()
    
    col_list = set(col_list)
    str_cols = col_list & features_type(str(), data)
    float_cols = col_list & features_type(float(), data)
    int_cols = col_list & features_type(int(),data)
     
    print len(col_list), "features:", len(str_cols), "categorical features", len(int_cols), "ordinal features", len(float_cols), "Numeric features"
    print
    
    # Categorical features
    if len(str_cols) > 0:
      print len(str_cols), "categorical features"
      print "-----------------------------------"
      print "{:<50s} {:8s}  {:50s}".format("Column Name", "# Unique", "Items")
      for col in str_cols:
          unique = data[col].unique()
          print "{:<50s} {:8d}  {:50s}".format(col, len(unique), sorted(unique))

    # Ordinal features 
    if len(int_cols) > 0:
        print
        print len(int_cols), "ordinal features"
        print "-------------------------------"
        print "{:<50s} {:8s}  {:>4s}  {:>4s} {:>5s}".format("Column Name", "# Unique", "Min", "Max", "Std")
        for col in int_cols:
           print  "{:<50s} {:8n}  {:>4n}  {:>4n} {:>5.2f}".format(col, len(data[col].unique()), data[col].min(), data[col].max(), data[col].std())
      
    # Numeric features 
    if len(float_cols) > 0:
        print
        print  len(float_cols), "Numeric features"
        print "----------------------------------"
        print "{:<50s} {:8s}  {:>4s}  {:>12s} {:>5s}".format("Column Name", "# Unique", "Min", "Max", "Std")
        for col in float_cols:
           print  "{:<50s} {:8d}  {:>12,.2f}  {:>12,.2f} {:>5,.2f}".format(col, len(data[col].unique()), data[col].min(), data[col].max(), data[col].std())
      
def show(col, data):     
    col_type = data[col].dtype()
    if col_type == type(str()):
        print "{:<50s}  {:18s} {:8d}  {:30s}". format(col, col_type, len(data[col].unique()), sorted(data[col].unique()))
    else:
        print "{:<50s}  {:18s} {:8d}  {:12.0f}  {:12.0f} {:12.0f}". format(col, col_type, len(data[col].unique()), data[col].min(), data[col].max(), data[col].std())
        
def print_heading():
            print "{:<50s}  {:18s} {:8s}  {:>12s}  {:>12s} {:>12s}". format("Column Name", "type", "# unique", "min", "max", "std")
    
    
    
def mode_sa(sa, single_mode=True):
    """Return a mode of sa, or all modes if there are several.
    single_mode: whether to return a single mode or a list of all modes (default: True)."""

    sf = gl.SFrame({"value": sa})
    sf2 = sf.groupby("value", {"count": gl.aggregate.COUNT()})
    max_count_index = sf2["count"].argmax()

    if single_mode:
        return sf2[max_count_index]["value"]

    else:
        max_count = sf2[max_count_index]["count"]
        return sf2[sf2["count"] == max_count]["value"]

def frequency(sa):    
    '''SFrame of the frequencies of the data in the SArray sa'''
  
    sf = gl.SFrame({"Value": sa})
    sf2 = sf.groupby("Value", {"Freqency": gl.aggregate.COUNT()}).sort('Freqency', ascending=False)
    sf2['Freqency'] = sf2['Freqency']/len(sa)
    sf2['Freqency'] = sf2['Freqency'].apply(lambda x: "{:2.2%}".format(x))
    return sf2


def plot_metric(model):
    '''Plot training and validation meterics'''
    
    data=model['progress']
    fig = plt.figure()
    title = "{:s}: {:4.4f}".format(model['metric'], model['validation_auc'])
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.plot(data['Iteration'], data['Training-auc'],   'b.', label=['Train'])
    plt.plot(data['Iteration'], data['Validation-auc'], 'g^', label=['Validate'])
    plt.legend()
#    plt.hlines(model['validation_auc'], 0, data['Iteration'][-1])
    
    return plt

  
def plot_roc(model, data):
    '''Plot ROC'''
    
    roc = model.evaluate(data, metric='roc_curve')['roc_curve']
    fig = plt.figure()
    try:
        title = "{:s}: {:4.6f}".format(model['metric'], model['validation_auc'])
    except:
        title = "ROC Curve"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.plot(roc['fpr'], roc['tpr'])
    
    return fig
  

def make_submission(model, data, file):
    '''Create a submission in Submissions/file in the correct format'''
    
    tmp = gl.SFrame({'ID': data['ID']})
    tmp['TARGET'] = model.predict(data, output_type='probability')
    tmp['ID', 'TARGET'].save('Submissions/'+ file + '.csv')
    
    return


import numpy as np

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def PCA_sf(sf, n_components=None):
    '''Compute the PCA tranform on a SFrame'''
   
    np = sf.to_numpy()

    np = normalize(np, axis=0)
    pca = PCA(n_components=n_components, copy=True)
    np = pca.fit_transform(np)
    return np, pca.explained_variance_ratio_



def cov_sa(sa1, sa2):
    '''covariance'''
    n = float(len(sa1))
    
    sum1 = sa1.sum()
    sum2 = sa2.sum()
    sum12 = (sa1*sa2).sum()
    
    return (sum12 - sum1*sum2 / n) / n  


def pearsonr_sa(sa1, sa2):
    '''Pearson correlation of two SArrays'''
    
    std1=sa1.std()
    std2=sa2.std()
    if std1 == 0 or std2 == 0:
        return 0
    
    return cov_sa(sa1, sa2)/(std1*std2)

def stratified_split(data, target, fraction, seed=None):
    
    train = gl.SFrame()
    test = gl.SFrame()
    classes = data[target].unique()
    
    for i, c in enumerate(classes):
        data_c=data[data[target] == c]
        test_c, train_c = data_c.random_split(fraction, seed)

        test = test.append(test_c)
        train = train.append(train_c)

    #shuffle to mix up order of classes
    test = gl.cross_validation.shuffle(test)
    train = gl.cross_validation.shuffle(train)
        
    return test, train


from ipywidgets import IntProgress
from IPython.display import display
from time import sleep
from os import listdir

def progress_bar(job):
    '''Display a status bar showing how many tasks are completed'''
    
    status=get_status(job)
    f = IntProgress(min=0, max=status['Total'], bar_style='success')
    f.value = status['Total'] - status['Pending'] - status['Running']
    f.description =  "{:1.0f} tasks left ".format(status['Pending']+status['Running'])
    display(f)
    
    while f.value <  status['Total']:
        status=get_status(job)
        f.value = status['Total'] - status['Pending'] - status['Running']
        f.description =  "{:1.0f} tasks left ".format(status['Pending']+status['Running'])
        if status['Failed'] > 0:
            f.bar_style='warning'
        sleep(1)

        
def get_status(job):
        """
        Get the status of all jobs launched for this model search.
        """
        status = {'Total':0, 
                  'Completed': 0,
                  'Running'  : 0,
                  'Failed'   : 0,
                  'Pending'  : 0,
                  'Canceled' : 0}
        for j in job.jobs:
            job_status = j.get_status(_silent=True)
            status['Total'] += len(j._stages[0])
            
            if job_status == 'Completed':
                result = j.get_results()

                # Increment overall status with the map_job's status
                for k, v in result['status'].iteritems():
                    status[k] += v
            elif job_status == 'Running':
                # KMc - number of completed jobs is the number of output files less one on the basis that 
                # either one task is still actively writing to this output or the output file is the completion task. 
                status['Completed'] += max(0, len(listdir(j._exec_dir + "/output"))-1)
                status['Running'] += 1
                status['Pending'] +=  max(0,len(j._stages[0]) - max(0, len(listdir(j._exec_dir + "/output"))-1) - 1)
            else:
                # Otherwise assume all tasks have the same status as the job
                status[job_status] += len(j._stages[0])
                
        return status