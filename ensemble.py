import graphlab as gl
from graphlab.toolkits._main import ToolkitError
import os
import json
from collections import Counter

<<<<<<< HEAD

def load_model(location):

    if not os.path.exists(location):
        raise IOError(location + ' does not exist')

    with open(location+"/data.json", "r") as f:
        data = json.load(f)

    lst = [gl.load_model(location+"/"+f) for f in os.listdir(location) if f != 'data.json']
    
    return Ensemble(lst,  weights=data['weights'], vote_fn=data['vote_fn'])


# standard method to vote on class
def vote(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

=======
>>>>>>> e83941829d12b0e0062206a0e81a0eac4d9e9c26
# this should probably be a subclass of classifier
# but that's more than I can figure out now

class Ensemble:
  
    def __init__(self, lst, weights=None, vote_fn=vote):
        '''create an Esemble from a list of classifier models.
        
        weights: the weights used to belend each models prediciton probability.weights
                 None means equal weighting 
                 
        vote_fn: the function to decide which class is predicted
                 the default is most common prediction class'''
        # Should include error checking code to test that the elements of lst are classifier models
        self.lst = lst
        self.vote_fn = vote_fn 
            
        if weights == None:
            self.weights = [1/float(len(lst))]*len(lst)
        elif len(lst) == len(weights):  # Should check that weights is a list of floats that sums to 1
            self.weights = weights
        else:
            raise ValueError('Weights must be same length as lst')
            
    def evaluate(self, data, metric='accuracy',missing_value_action='auto'):
      
<<<<<<< HEAD
        if metric in ['accuracy', 'confusion_matrix', 'f1_score', 'fbeta_score',  'precision', 'recall']:
            output_type = 'class'
        elif metric in ['log_loss','roc_curve', 'auc']:
=======
        if metric in ['accuracy', 'confusion_matrix', 'f1_score', 'fbeta_score', 'log_loss', 'precision', 'recall']:
            output_type = 'class'
        elif metric in ['roc_curve', 'auc']:
>>>>>>> e83941829d12b0e0062206a0e81a0eac4d9e9c26
            output_type = 'probability'
        else:
          raise ToolkitError('Evaluation metric "' + metric + '" not recognized')
                
        targets = data[self.lst[0].target]
        pred = self.predict(data, output_type=output_type, missing_value_action=missing_value_action)
        
<<<<<<< HEAD
        return {metric: eval("gl.evaluation."+metric)(targets, pred)}
=======
        return eval("gl.evaluation."+metric)(targets, pred)
>>>>>>> e83941829d12b0e0062206a0e81a0eac4d9e9c26

          
        
    def predict(self, dataset, output_type='probability', missing_value_action='auto'):

        # standard method to vote on class
        def vote(lst):
            data = Counter(lst)
            return data.most_common(1)[0][0]
<<<<<<< HEAD
          
=======

>>>>>>> e83941829d12b0e0062206a0e81a0eac4d9e9c26
        if output_type == 'class':
            sf = gl.SFrame([m.predict(dataset, output_type, missing_value_action) for m in self.lst])
            #This raises an import error, so hard code it for now
            #return sf.apply(lambda row: self.vote_fn(row.values()))
            return sf.apply(lambda row: vote(row.values()))
        elif output_type == 'probability':
            return sum([m.predict(dataset, output_type, missing_value_action)*w for m,w in zip(self.lst, self.weights)])

            
    def len(self):
        return len(self.lst)
      
    def __str__(self):
<<<<<<< HEAD
        s = str({'weights': self.weights, 'vote_fn':self.vote_fn}) + "\n"
        s += str([m.__name__ for m in self.lst])
        return s
=======
        return str([m.__name__ for m in self.lst])
>>>>>>> e83941829d12b0e0062206a0e81a0eac4d9e9c26
      

    def save(self, location):

        if not os.path.exists(location):
            os.makedirs(location)

        for i, m in enumerate(self.lst):
            m.save(location + "/model_" + str(i))

        data ={'weights': self.weights, 'vote_fn': self.vote_fn.func_name}
        with open(location+"/data.json", "w") as f:
            json.dump(data, f)    
    
<<<<<<< HEAD
=======

def load_model(location):

    if not os.path.exists(location):
        raise IOError(location + ' does not exist')

    with open(location+"/data.json", "r") as f:
        data = json.load(f)

    lst = [gl.load_model(location+"/"+f) for f in os.listdir(location) if f != 'data.json']
    
    return Ensemble(lst,  weights=data['weights'], vote_fn=data['vote_fn'])


  
>>>>>>> e83941829d12b0e0062206a0e81a0eac4d9e9c26
