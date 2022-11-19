#imports
from models.random_forest.abstract.base_randomforest import RandomForest
from models.decisiontrees import DecisionTreeClassifier
import numpy as np
import pandas as pd

#class for random forest classifier
class RandomForestClassifier(RandomForest):
    #initializer
    def __init__(self,n_trees=100,max_depth=None,min_samples_split=2,loss='gini'):
        super().__init__(n_trees)
        self.max_depth             = max_depth
        self.min_samples_split     = min_samples_split
        self.loss                  = loss
        
    #protected function to obtain the right decision tree
    def _make_tree_model(self):
        return(DecisionTreeClassifier(self.max_depth, self.min_samples_split, self.loss))
    
    #public function to return model parameters
    def get_params(self, deep = False):
        return {'n_trees':self.n_trees,
                'max_depth':self.max_depth,
                'min_samples_split':self.min_samples_split,
                'loss':self.loss}
    
    #train the ensemble
    def fit(self,X_train,y_train):
        #call the protected training method
        dcOob = self._train(X_train,y_train)
            
    #predict from the ensemble
    def predict(self,X):
        #call the protected prediction method
        ypred = self._predict(X)
        #return the results
        return(ypred)  

    ### Updated by Tomoki Kyotani
    def find_important_features(self, num_indexes):

        feature_importances = { i:0 for i in range(num_indexes) }
        
        for tree in self.trees:
            level = 1
            feature_importances = self.find_important_features_helper(tree.tree, level, feature_importances)
        
        return feature_importances

    def find_important_features_helper(self, n, level, feature_importances):
        if level <= 3:
            if level == 1:
                imporance_score = 3
            elif level == 2:
                imporance_score = 2
            else:
                imporance_score = 1

            feature_importances[n.feature_index] += imporance_score
            self.find_important_features_helper(n.get_left_node(), level+1, feature_importances)
            self.find_important_features_helper(n.get_right_node(), level+1, feature_importances)
            return feature_importances