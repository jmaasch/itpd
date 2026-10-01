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
from string import ascii_uppercase as alphabet

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

# Graph metrics.
#from dodiscover.metrics import structure_hamming_dist as SHD
from cdt.metrics import precision_recall
from cdt.metrics import SHD
from cdt.metrics import SID

# Custom scripts.
from padl_naive import PaDL


class ITPDNaive:

    def __init__(self,
                 df: pd.DataFrame,
                 var_names: list,
                 pairs: list[tuple] = None,
                 independence_test: str = "fisherz",
                 alpha: float = 0.01,
                 tau_max: int = None,
                 true_adj: nx.DiGraph = None):

        self.df = df
        self.var_names = var_names
        self.pairs = pairs
        self.true_adj = true_adj
        self.M = self.df.shape[0]
        self.N = len(self.var_names)
        self.T = len(self.df.columns) // self.N

        # Init results storage.
        self.label_dicts = dict()
        self.total_tests = []
        self.conditioning_set_sizes = []

        # Utils.
        self.u = Utils()

        # Maximum lag to consider for parent search.
        # Default = None (full history).
        self.tau_max = tau_max

        # Instantiate objects for independence testing.
        self.test_name = independence_test
        if independence_test != "oracle":
            self.test = CIT(self.df.to_numpy(), independence_test)
        else:
            self.test = "oracle"
        self.alpha = alpha


    def itpd_naive(self,
                   verbose: bool = False):

        '''
        Run vanilla PaDL iteratively.
        '''

        #print("var names:", self.var_names)
        #print(f"M x T x N = {self.M} x {self.T} x {self.N}: ")
        
        if self.pairs is None:
            self.pairs = []
            for t in range(self.T-1):
                for var in self.var_names:
                    self.pairs.append((var+"_"+str(t), var+"_"+str(t+1)))

        #print("self.pairs", self.pairs)

        # Result storage.
        self.c2p_pred = dict() # child : parents
        self.p2c_pred = {pair[0] : [pair[1]] for pair in self.pairs} # parent : children

        #display(self.p2c_pred)
        
        start = time.time()
        for pair in self.pairs:
            #if verbose:
            #print("\n", pair)
        
            # Drop variables from dataframe (respect temporal order).
            t = int(pair[1].split("_")[-1])
            drop_vars = [x+"_"+str(i) for x in self.var_names for i in range(t,self.T)]
            drop_vars = [x for x in drop_vars if x not in pair]
            if self.tau_max is not None:
                drop_vars += [x+"_"+str(i) for x in self.var_names for i in range(0,t-self.tau_max)]
            df_drop = self.df.drop(columns = drop_vars)
            #print("candidates", df_drop.columns)
        
            # Drop variables from adjacency matrix.
            if self.true_adj is not None:
                adj_idx = [list(self.df.columns).index(x) for x in df_drop.columns]
                true_adj_dropped = self.true_adj[adj_idx][:,adj_idx]
            else:
                true_adj_dropped = None
        
            pa, labels, total_tests, cond_sizes = self.run_padl(
                df = df_drop,
                exposure = pair[0],
                outcome = pair[1],
                true_adj = true_adj_dropped,
                verbose = verbose
            )

            # Store results.
            self.c2p_pred[pair[1]] = pa
            for p in pa:
                self.p2c_pred[p] += [pair[1]]
            self.label_dicts[pair] = labels
            self.total_tests.append(total_tests)
            self.conditioning_set_sizes.append(cond_sizes)

            #if verbose:
            #    print("\n-----------------------------\n")
            
        self.runtime = time.time()-start
        print(f"ITPD complete in {round(self.runtime,4)} seconds.")
        
        # Convert to indices.
        # Get predicted adjacency matrix.
        p2c_pred_idx = dict()
        cols = list(self.df.columns)
        for pa,ch in self.p2c_pred.items():
            p2c_pred_idx[cols.index(pa)] = [cols.index(c) for c in ch]
        self.itpd_adj = self.u.make_adjacency_matrix((len(cols),len(cols)), p2c_pred_idx)
        
        # Score.
        if self.true_adj is not None:
            aupr, shd = self.u.score_arrays(self.true_adj, self.itpd_adj)

            if verbose:
                # Sanity check on total edges.
                print("\nTotal true edges:", np.sum(self.true_adj))
                print("Total predicted edges:", np.sum(self.itpd_adj))
    
                # Scores.
                print(f"Area under precision/recall curve: {aupr[0]}")
                print(f"SHD: {shd}")
            return aupr[0], shd


    def run_padl(self,
                 df: pd.DataFrame = None,
                 exposure: str = "X",
                 outcome: str = "Y",
                 true_adj: np.array = None,
                 verbose: bool = False) -> tuple(list,dict,int,list):

        if df is None:
            df = self.df
        padl = PaDL(data = df, 
                    independence_test = self.test_name)
        if true_adj is not None:
            padl.dag = true_adj
            padl.var_names = list(df.columns)
        
        start = time.time()
        pa = padl.get_parents(exposure = exposure, 
                              outcome = outcome,
                              alpha = self.alpha)
        if verbose:
            print("\nResults complete in {}s.".format(round(time.time() - start, 4)))
            print("Total independence tests performed:", padl.total_tests)
            print("Predicted parents of outcome =", sorted(pa))

        return pa, padl.pred_label_dict, padl.total_tests, padl.conditioning_set_sizes


class Utils:


    def get_data(self,
                 M: int = 100, 
                 T: int = 10,
                 N: int = 5, 
                 max_children: int = 3,
                 plot: bool = True) -> tuple(pd.DataFrame, dict, dict, nx.DiGraph, np.array):
    
        '''
        Generate a random nonstationary time series. Data will ultimately be M x T x N.
        - Gaussian noise.
        - No causal stationarity.
        - Linear causal mechanisms.
    
        
        Params:
        --------
        M: total replicate trajectories
        T: total time steps
        N: total variables
        max_children: maximum number of children per node
        plot: plot causal DAG
    
        Return:
        --------
        df, p2c, c2p, G, true_adj
        '''

        if N > len(alphabet):
            raise ValueError(f"N cannot be greater than {len(alphabet)}")
        var_names = list(alphabet[:N])
    
        # Parent : children dictionary.
        # All variables are autocorrelated.
        p2c = {v+"_"+str(i) : [v+"_"+str(i+1)] for v in var_names for i in range(T)}
        var_names_full = list(p2c.keys())
    
        # Empty out final time step (leaves).
        for v in var_names:
            p2c[v+"_"+str(T-1)] = []
    
        # Randomly select additional children.
        n_children = lambda : np.random.choice(range(max_children+1), size = 1)[0]
        for v in var_names:
            for t in range(T-1):
                candidates = [x+"_"+str(i) for x in var_names for i in range(t+1,T) if x+"_"+str(i) not in (v+str(t),v+str(t+1))]
                p2c[v+str(t)] += np.random.choice(candidates, size = n_children()).tolist()
    
        # Child : parents dictionary.
        c2p = {p : [] for p in p2c.keys()}
        for parent,children in p2c.items():
            for child in children:
                c2p[child] += [parent]
    
        # Generate graph.
        G = nx.from_dict_of_lists(p2c, create_using = nx.DiGraph)
        true_adj = nx.to_numpy_array(G)
        if plot:
            self.plot_nx(true_adj, labels = list(p2c.keys()))
        
        # Generate data.
        # Data will ultimately be m x t x n:
        data_dict = {var : np.random.normal(size = M) for var in var_names_full}
        c = lambda : np.random.choice([-0.8, -1.5, 1.5, 0.8], size = 1)[0]
        for var in var_names_full:
            for parent in c2p[var]:
                data_dict[var] += c() * data_dict[parent]
        df = pd.DataFrame(data_dict)       
    
        return df, p2c, c2p, G, true_adj


    def format_tigramite(self,
                         df: pd.DataFrame,
                         var_names: list[str]) -> tupe(pd.DataFrame,np.array):

        mxt = []
        m_dict = dict()
        for var in var_names:
            # M x T
            var_data_mxt = df[[x for x in df.columns if var in x]].to_numpy()
            mxt.append(var_data_mxt)

        # For m in M rows, concat m^th row of each array into one column vec.
        for i in range(mxt[0].shape[0]):
            current_array = []
            # For each of N vars, add row i to current array.
            for j in range(len(var_names)):
                current_array.append(mxt[j][i].reshape(-1,1))
            m_dict[i] = np.column_stack(current_array)

        return m_dict

        
    def plot_nx(self,
                adjacency_matrix: np.array,
                labels = None,
                figsize = (5,5),
                dpi = 75,
                node_size = 800,
                arrow_size = 10):
        
        if labels is None:
            labels = [i for i in range(adjacency_matrix.shape[0])]
            
        g = nx.from_numpy_array(adjacency_matrix, create_using = nx.DiGraph)
        plt.figure(figsize = figsize, dpi = dpi)  
        nx.draw_circular(g, 
                         node_size = node_size, 
                         node_color = "pink",
                         labels = dict(zip(list(range(len(labels))), labels)), 
                         arrowsize = arrow_size,
                         with_labels = True)
        plt.show()
        plt.close()


    def make_adjacency_matrix(self,
                              size: tuple, 
                              parents: dict) -> np.array:
    
        '''
        parents: dictionary of lists, as used in constructor nx.from_dict_of_lists
            parents[i] = [j0, ... , jk] for k children of i
            i,j are integers (indices)
    
        Adjacency matrices in networkx: 
            for directed graphs, entry i, j corresponds to an edge from i to j.
        '''
    
        adj = np.zeros(size)
        for parent,children in parents.items():
            adj[parent,children] = 1
        return adj


    def get_parents(self,
                    adj: np.array) -> dict:
        
        '''
        Params
        -------
        adj: array
            Adjacency matrices in networkx: 
            for directed graphs, entry i, j corresponds to an edge from i to j.
        
        Return
        -------
        parents: dictionary of lists, as used in constructor nx.from_dict_of_lists
            parents[j] = [i0, ... , ik] for k children of j
            i,j are integers (indices)
        '''
    
        parents = dict()
        for i in range(adj.shape[0]):
            parents[i] = np.nonzero(adj[i,:])[0].tolist()
        return parents
        

    def score_arrays(self,
                     true_adj: np.array, 
                     pred_adj: np.array) -> tuple: 

        '''
        Score area under the precision-recall curve (AUPRC) and 
        structural hamming distance for the predicted adjacency 
        matrix relative to ground truth.

        AUPRC from: 
        https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/metrics.html
        '''
    
        pr = precision_recall(true_adj, pred_adj)
        shd = SHD(true_adj, pred_adj)
        
        return pr, shd


