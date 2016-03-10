import os as os
import graphlab as gl


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