import os as os
import graphlab as gl
import tools 
import csv

def translate_columns(train_data,test_data):
    '''Translate column names from Spanish to English'''
  
    with open('Data/Spanish2English.csv', 'rb') as f:
        reader = csv.reader(f)
        headers = reader.next()
        spanish2english = dict(reader)

    print len(spanish2english) ,"Columns translated to English"
    train_data.rename(spanish2english)
    test_data.rename(spanish2english)
    
    return train_data,test_data

def delete_redundant_columns(train_data,test_data):
    '''Remove columns that are constant or duplicates'''
  
    #These are calculated in the data discovery notebook
    constant_cols = ['var2 indicator 0', 'var2 indicator', 'var27 indicator 0', 'var28 indicator 0', 'var28 indicator', 'var27 indicator', 'var41 indicator', 'var46 indicator 0', 'var46 indicator', 'var27 number 0', 'var28 number 0', 'var28 number', 'var27 number', 'var41 number', 'var46 number 0', 'var46 number', 'var28 balance', 'var27 balance', 'var41 balance', 'var46 balance', 'var18 repayment amount hace3', 'var34 repayment amount hace3', 'var13 refund amount hace3', 'var33 refund amount hace3', 'var17 transfer amount out hace3', 'var33 transfer amount  out hace3', 'var2 number 0 ultima1', 'var2 number ultima1', 'var13 reimbursement number hace3', 'var33 reimbursement number hace3', 'var17 number transfer out hace3', 'var33 number transfer out hace3', 'var2 balance ultima1', 'var13 average balance means hace3']
    duplicate_cols = ['var29 indicator 0', 'var29 indicator', 'var13 indicator means', 'var18 indicator', 'var26 indicator', 'var25 indicator', 'var32 indicator', 'var34 indicator', 'var37 indicator', 'var39 indicator', 'var29 number 0', 'var29 number', 'var13 average number', 'var18 number', 'var26 number', 'var25 number', 'var32 number', 'var34 number', 'var37 number', 'var39 number', 'var29 balance', 'var13 average balance means ultima1', 'var33 delta contribution amount 1Y3', 'var13 delta reimbursement amount 1Y3', 'var17 delta reimbursement amount 1Y3', 'var33 delta reimbursement amount 1Y3', 'var17 delta transfer number in 1Y3', 'var17 delta transfer number out 1Y3', 'var33 delta transfer number in 1Y3', 'var33 delta transfer number out 1Y3']
    drop_col = set(constant_cols + duplicate_cols)
  
    print len(drop_col) ,"constant or duplicate columns removed"
    train_data.remove_columns(drop_col);
    test_data.remove_columns(drop_col);
  
    return train_data,test_data

def clean_data(train_data,test_data):
    '''features with anomalous values (e.g., -999999, 9999999999)or replaced by 'None'''  
    col_maxed = [col for col in train_data.column_names() if train_data[col].max() == 9999999999]
    print len(col_maxed)+1, "features with anomalous values replaced by 'None'" 
    
    for data in [train_data, test_data]:
        data['var3'] = subna(data['var3'], -999999)
        for col in col_maxed:
            data[col] = subna(data[col], 9999999999)

    return train_data, test_data 
    
def subna(data, value):
    '''Replaced value in data with None'''
    return data.apply(lambda i: None if i== value else i)
  
def convert_categorical_features(train_data,test_data):
    '''Convert features like "indicator" or "delta" to categorical  (i.e., strings)'''

    categoricals = tools.features_like(".*indicator", train_data) | tools.features_like(".*delta", train_data)
    
    print len(categoricals), "integer features converted to categorical"
    for col in categoricals:
        train_data[col] = train_data[col].astype(str)
        test_data[col] = test_data[col].astype(str)

    train_data['TARGET'] = train_data['TARGET'].apply(lambda i: "Satisified" if i == 1 else 'Unsatisified')
     
    return train_data, test_data         
  
  
def prepare_data(categorical=True, clean=True):
    '''Load and prepare data, return as train and test tupple'''

# new training and testing data sets
    print "Loading raw data from CSV files"
    train_data     = gl.SFrame.read_csv('Data/train.csv', verbose=False)
    test_data      = gl.SFrame.read_csv('Data/test.csv', verbose=False)
    
    print "train:", train_data.num_rows(), "   test:",  test_data.num_rows()
    print len(train_data.column_names()) - 1, "raw features"

    train_data, test_data = translate_columns(train_data, test_data)
    train_data, test_data = delete_redundant_columns(train_data, test_data)
    if clean:
        train_data, test_data = clean_data(train_data,test_data)
    if categorical:
        train_data, test_data = convert_categorical_features(train_data,test_data)

    print len(train_data.column_names()) - 1, "features in total"
    
    return train_data, test_data
  

def load_data(reload_data=False):  
    '''Load transformed and merged data'''
        
    TEST_CACHE  = 'Data/train.gl'
    TRAIN_CACHE = 'Data/test.gl'
    
    if reload_data or not os.path.exists(TRAIN_CACHE) or not os.path.exists(TEST_CACHE) :
        print 'Loading raw data and tranforming it'
        train, test = prepare_data()
        
        print "Saving processed data for fast reloading"
        train.save(TRAIN_CACHE)
        test.save(TEST_CACHE)

    else:
        print 'Loading saved processed data'
        train = gl.SFrame(TRAIN_CACHE)
        test = gl.SFrame(TEST_CACHE)
        print len(train.column_names()) - 1, " features in total"
        
    return train, test