# General imports.
import pandas as pd
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
import math
from scipy import stats

# Independence testing imports.
from causallearn.utils.cit import CIT
import networkx as nx

# sklearn imports.
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


class PaDL():

    def __init__(self,
                 data: pd.DataFrame = None,
                 independence_test: str = "fisherz"):

        '''
        independence_test in ["chisq", "fisherz", "kci", "gsq", "oracle"]
        '''

        # Store data.
        self.data = data
        self.var_names = list(data.columns)

        # Instantiate objects for independence testing.
        if independence_test != "oracle":
            self.test = CIT(self.data.to_numpy(), independence_test)
        else:
            self.test = "oracle"

        # Init data structures for memoization.
        self.x_ind_z_dict = dict()
        self.y_ind_z_dict = dict()

        # Track how many tests were performed.
        self.total_tests = 0
        self.conditioning_set_sizes = []
        

    def get_parents(self,
                    exposure: str = "X",
                    outcome: str = "Y",
                    alpha: float = 0.005,
                    verbose: bool = False) -> dict:

        # Extract variable names from columns.
        if self.data is not None:
            self.z_names = list(self.data.columns)
        elif self.var_names is not None:
            self.z_names = self.var_names.copy()
        else:
            raise ValueError("self.data and self.var_names cannot both be None.")
        self.z_names.remove(exposure)
        self.z_names.remove(outcome)

        # Initialize data structures that will store results.
        self.pred_label_dict  = dict()
        self.z_prime          = []
        self.z1_z3            = [] # Confounders and mediators that are parents of outcome.
        self.z4               = [] # Non-descendants of outcome.
        self.z4_parents       = [] # Z4 that are parents of outcome.
        self.z7               = [] # Children of exposure.
        self.z8               = [] # Isolated variables.
        self.z_not_adj_y      = [] # Unlabeled variables not adjacent to outcome.

        for candidate in self.z_names:

            #---------------------
            # Step 1: Test for Z8.
            #---------------------
            
            z8 = self.test_z8(exposure = exposure,
                              outcome = outcome,
                              candidate = candidate,
                              alpha = alpha,
                              verbose = verbose)
            if z8:
                self.z8.append(candidate)
                self.pred_label_dict[candidate] = "Z8"
                continue

            #---------------------
            # Step 2: Test for Z7.
            #---------------------

            z7 = self.test_z7(exposure = exposure,
                              outcome = outcome,
                              candidate = candidate,
                              alpha = alpha,
                              verbose = verbose)
            if z7:
                self.z7.append(candidate)
                self.pred_label_dict[candidate] = "Z7"
                continue

            #---------------------
            # Step 3: Test for Z4.
            #---------------------

            z4 = self.test_z4(exposure = exposure,
                              outcome = outcome,
                              candidate = candidate,
                              alpha = alpha,
                              verbose = verbose)
            if z4:
                self.z4.append(candidate)
                self.pred_label_dict[candidate] = "Z4"
                continue
                

            # If no tests pass, add to z_prime.
            self.z_prime.append(candidate)
        
        #------------------------------
        # Step 4: Z1,3 Adjacent to Y.
        #------------------------------

        for candidate in self.z_prime:
        
            adj = self.test_adj_y(exposure = exposure,
                                  outcome = outcome,
                                  candidate = candidate,
                                  alpha = alpha,
                                  verbose = verbose)
            if adj:
                self.z1_z3.append(candidate)
                self.pred_label_dict[candidate] = "Z1 or Z3"
            else:
                self.z_not_adj_y.append(candidate)
                #self.pred_label_dict[candidate] = "Not adjacent to Y"
                self.pred_label_dict[candidate] = "Not identifiable"

        #------------------------------
        # Step 5: Z4 adjacent to Y.
        #------------------------------

        for z4 in self.z4:
            conditioning_set = [x for x in self.z4 if x != z4] + self.z1_z3 + [exposure]
            self.conditioning_set_sizes.append(len(conditioning_set))
            p_value = self.ind_test(var_0 = z4, 
                                    var_1 = outcome, 
                                    conditioning_set = conditioning_set)
            if p_value <= alpha:
                self.z4_parents.append(z4)

        # Return all identifiable parents.
        return self.z1_z3 + self.z4_parents


    def ind_test(self,
                 var_0: str,
                 var_1: str,
                 conditioning_set: list = None) -> float:

        '''
        Wrapper for independence test that takes variable names as strings
        for easier user interpretability.

        Return:
        -------
        p_value
        '''

        if self.test != "oracle":
            # Obtain column indices (independence test references array columns).
            if self.data is None:
                raise ValueError("No data provided. Set self.data to pandas dataframe to proceed.")
            df_cols = list(self.data.columns)
            var_0_idx = df_cols.index(var_0)
            var_1_idx = df_cols.index(var_1)
            if conditioning_set is not None:
                cond_idx = [df_cols.index(x) for x in conditioning_set]
            else:
                cond_idx = None
            p_value = self.test(var_0_idx, var_1_idx, cond_idx)
        else:
            # Oracle test using ground truth DAG.
            p_value = self.oracle(var_0, var_1, conditioning_set = conditioning_set)

        # Increment total tests performed.
        self.total_tests += 1

        return p_value


    def oracle(self,
               var_0,
               var_1,
               conditioning_set = None) -> float:

        '''
        Oracle independence test given ground truth DAG.
        '''

        if self.dag is None:
            raise ValueError("self.dag is None; must supply ground truth DAG as numpy array to use oracle.")
        
        # Obtain column indices.
        var_0_idx = self.var_names.index(var_0)
        var_1_idx = self.var_names.index(var_1)
        if conditioning_set is not None:
            if var_0 in conditioning_set:
                conditioning_set.remove(var_0)
            if var_1 in conditioning_set:
                conditioning_set.remove(var_1)
            cond_idx = set([self.var_names.index(x) for x in conditioning_set])
        else:
            cond_idx = set()

        # Get p-value using ground truth DAG.
        graph = nx.from_numpy_array(self.dag, create_using = nx.DiGraph)
        p_val = 1 if nx.is_d_separator(graph, {var_0_idx}, {var_1_idx}, cond_idx) else 0
        return p_val


    def test_z8(self,
                exposure: str = "X",
                outcome: str = "Y",
                candidate: str = "Z",
                alpha: float = 0.005,
                verbose: bool = False) -> bool:

        # Test marginal independence of X and Z.
        if candidate in self.x_ind_z_dict:
            x_ind_z = self.x_ind_z_dict.get(candidate)
        else:
            x_ind_z = self.ind_test(var_0 = exposure,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.x_ind_z_dict[candidate] = x_ind_z

        # Test marginal independence of Y and Z.
        if candidate in self.y_ind_z_dict:
            y_ind_z = self.y_ind_z_dict.get(candidate)
        else:
            y_ind_z = self.ind_test(var_0 = outcome,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.y_ind_z_dict[candidate] = y_ind_z

        #print("Z8 test for candidate {} | x_ind_z: {} | y_ind_z: {}".format(candidate, x_ind_z, y_ind_z))

        # Test for Case 8.
        #if x_ind_z and y_ind_z:
        if x_ind_z > alpha and y_ind_z > alpha:
            return True
        return False

    
    def test_z7(self,
                exposure: str = "X",
                outcome: str = "Y",
                candidate: str = "Z",
                alpha: float = 0.005,
                verbose: bool = False) -> bool:

        # Test marginal independence of Y and Z.
        if candidate in self.y_ind_z_dict:
            y_ind_z = self.y_ind_z_dict.get(candidate)
        else:
            y_ind_z = self.ind_test(var_0 = outcome,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.y_ind_z_dict[candidate] = y_ind_z

        # Test conditional independence of Y and Z given X.
        y_ind_z_given_x = self.ind_test(var_0 = outcome,
                                        var_1 = candidate,
                                        conditioning_set = [exposure])

        # Marginally dependent and conditionally independent.
        if y_ind_z <= alpha and y_ind_z_given_x > alpha:
            return True
        return False


    def test_z4(self,
                 exposure: str = "X",
                 outcome: str = "Y",
                 candidate: str = "Z",
                 alpha: float = 0.005,
                 verbose: bool = False) -> bool:

        # Test marginal independence of X and Z.
        if candidate in self.x_ind_z_dict:
            x_ind_z = self.x_ind_z_dict.get(candidate)
        else:
            x_ind_z = self.ind_test(var_0 = exposure,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.x_ind_z_dict[candidate] = x_ind_z

        # Test conditional independence of X and Z given Y.
        x_ind_z_given_y = self.ind_test(var_0 = exposure,
                                        var_1 = candidate,
                                        conditioning_set = [outcome])

        # Test for Z4, which induces a v-structure X -> Y <- Z4.
        if x_ind_z > alpha and x_ind_z_given_y <= alpha:
            return True
        return False


    def test_adj_y(self,
                   exposure: str = "X",
                   outcome: str = "Y",
                   candidate: str = "Z",
                   alpha: float = 0.005,
                   verbose: bool = False) -> bool:

        # Test marginal independence of X and Z.
        if candidate in self.x_ind_z_dict:
            x_ind_z = self.x_ind_z_dict.get(candidate)
        else:
            x_ind_z = self.ind_test(var_0 = exposure,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.x_ind_z_dict[candidate] = x_ind_z

        # Test marginal independence of Y and Z.
        if candidate in self.y_ind_z_dict:
            y_ind_z = self.y_ind_z_dict.get(candidate)
        else:
            y_ind_z = self.ind_test(var_0 = outcome,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.y_ind_z_dict[candidate] = y_ind_z

        # Test conditional independence to assess adjacency.
        conditioning_set = self.z_prime + self.z4 + [exposure]
        self.conditioning_set_sizes.append(len(conditioning_set))
        if candidate in conditioning_set:
            conditioning_set.remove(candidate)
        #print("candidate:", candidate)
        #print("conditioning_set:", conditioning_set)
        z_cind_y = self.ind_test(var_0 = outcome,
                                 var_1 = candidate,
                                 conditioning_set = conditioning_set)
        #print("z_ind_y:", z_ind_y)

        if x_ind_z <= alpha and y_ind_z <= alpha and z_cind_y <= alpha:
            return True
        return False












