import numpy as np

# You are not allowed to import any libraries other than numpy

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF PROHIBITED LIBRARIES WILL RESULT IN PENALTIES

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc

################################
# Non Editable Region Starting #
################################
def my_fit( words, verbose = False ):
################################
#  Non Editable Region Ending  #
################################   
    
    # Use this method to train your decision tree model using the word list provided
	# Return the trained model as is -- do not compress it using pickle etc
	# Model packing or compression will cause evaluation failure
 
    dt = Tree( min_leaf_size = 1, max_depth = 15 )
    dt.fit( words, verbose )
    return dt                   # Return the trained model


class Tree:
    def __init__( self, min_leaf_size, max_depth ):
        self.root = None
        self.words = None
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
    
    def fit( self, words, verbose = False ):
        self.words = words
        self.root = Node( depth = 0, parent = None )
        if verbose:
            print( "root" )
            print( "└───", end = '' )
        self.root.fit( all_words = self.words, my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth, verbose = verbose )


class Node:
   
    def __init__( self, depth, parent ):
        self.depth = depth
        self.parent = parent
        self.all_words = None
        self.my_words_idx = None
        self.children = {}
        self.is_leaf = True
        self.query_idx = None
        self.history = []
        
    def get_query( self ):
        return self.query_idx

    def get_child( self, response ):
      
        if self.is_leaf:
            print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
            child = self
        else:         
            if response not in self.children:
                print( f"Unknown response {response} -- need to fix the model" )
                response = list(self.children.keys())[0]
            child = self.children[ response ]
        return child

    
    def process_leaf( self, my_words_idx, history ):
        return my_words_idx[0]

    def reveal( self, word, query ):
        mask = [ *( '_' * len( word ) ) ]
        for i in range( min( len( word ), len( query ) ) ):
            if word[i] == query[i]:
                mask[i] = word[i]
        return ' '.join( mask )
    
    def process_node( self, all_words, my_words_idx, history, verbose ):
        
        if len( history ) == 0:
            query_idx = -1
            query = ""
        else:
            query_idx = self.get_query_idx( all_words, my_words_idx)
            query = all_words[ query_idx ]
        split_dict = {}
        for idx in my_words_idx:
            mask = self.reveal( all_words[ idx ], query )
            if mask not in split_dict:
                split_dict[ mask ] = []
            split_dict[ mask ].append( idx )
        if len( split_dict.items() ) < 2 and verbose:
            print( "Warning: did not make any meaningful split with this query!" )
        return ( query_idx, split_dict )

    def fit( self, all_words, my_words_idx, min_leaf_size, max_depth, fmt_str = "    ", verbose = False ):
        self.all_words = all_words
        self.my_words_idx = my_words_idx
        if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
            self.is_leaf = True
            self.query_idx = self.process_leaf( self.my_words_idx, self.history )
            if verbose:
                print( '█' )
        else:
            self.is_leaf = False
            ( self.query_idx, split_dict ) = self.process_node( self.all_words, self.my_words_idx, self.history, verbose )
            if verbose:
                print( all_words[ self.query_idx ] )
            for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
                if verbose:
                    if i == len( split_dict ) - 1:
                        print( fmt_str + "└───", end = '' )
                        fmt_str += "    "
                    else:
                        print( fmt_str + "├───", end = '' )
                        fmt_str += "│   "
                self.children[ response ] = Node( depth = self.depth + 1, parent = self )
                history = self.history.copy()
                history.append( [ self.query_idx, response ] )
                self.children[ response ].history = history
                self.children[ response ].fit( self.all_words, split, min_leaf_size, max_depth, fmt_str, verbose )
                
    def get_query_idx(self,all_words, my_words_idx):
        bst_ent=np.inf
        query_idx=-1
        for i in my_words_idx:
            ent = self.try_idx(all_words, my_words_idx,i)
            if ent<bst_ent:
                query_idx=i
                bst_ent=ent
        return query_idx
    
    def try_idx(self,all_words,my_words_idx,i):
        count_dict = {}
        query=all_words[i]
        for idx in my_words_idx:
            mask = self.reveal( all_words[ idx ], query )
            if mask not in count_dict:
                count_dict[ mask ] = 0
            count_dict[ mask ]+=1
        return self.entropy( np.array( list( count_dict.values() ) ) )
    
    def get_entropy(self, counts ):    
        assert np.min( counts ) > 0, "Elements with zero or negative counts detected"
        num_elements = counts.sum()
        if num_elements <= 1:
            print( f"warning: { num_elements } elements in total." )
            return 0
        proportions = counts / num_elements
        return np.sum( proportions * -np.log2( 1/counts ) )
    
    def get_gini(self,counts):
        assert np.min( counts ) > 0, "Elements with zero or negative counts detected"
        num_elements = counts.sum()
        if num_elements <= 1:
            print( f"warning: { num_elements } elements in total." )
            return 0
        proportions = counts / num_elements    
        return np.sum( proportions * (1- 1/counts ) )
    
    def inv_gain_ratio(self,counts):
        assert np.min( counts ) > 0, "Elements with zero or negative counts detected"
        num_elements = counts.sum()
        if num_elements <= 1:
            print( f"warning: { num_elements } elements in total." )
            return 0
        proportions = counts / num_elements
        info_gain = -np.log2(num_elements)-np.sum( proportions * -np.log2( 1/counts ) )
        in_info_gain = np.sum( proportions * -np.log2( proportions ) )
        return in_info_gain/info_gain
