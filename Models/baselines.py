import math
import numpy as np

from numpy.linalg import norm
from Levenshtein import distance
from dataReader import processLabel
from pylcs import lcs_sequence_length

def normalizedLevenshtein(str1, str2):
    """
    Calculates the Levenshtein distance between two strings, normalized for the length of the longest string.
    |
    str1 (Str):     String 1
    str2 (Str):     String 2
    |
    output (Int)
    """
    dist = distance(str1.lower(), str2.lower())
    normalized = 1 - (dist / max(len(str1), len(str2)))
    return normalized

def longestCommonSubSequence(str1, str2):
    """
    Calculates the longest common subsequence between two strings, normalized for the length of the two strings.
    |
    str1 (Str):     String 1
    str2 (Str):     String 2
    |
    output (Int)
    """
    length = lcs_sequence_length(str1, str2)
    normilized = (length / (len(str1) + len(str2)))
    return normilized

def word2Vec(str1, str2, model, vec_size):
    """
    Calculates the cosine similarity between two strings based on vectors created by a Word2Vec model.
    |
    str1 (Str):             String 1
    str2 (Str):             String 2
    model (gensim.Model):   Word2Vec model
    dict (Dictionary):      Dictionary with precalculated vectors of the ontologies
    |
    output (Int)
    """
    vecs = []
    label1 = str1.split()
    label2 = str2.split()
    # For each label vectorize each word and combine the vectors
    for labels in [label1,label2]:
        vec = np.zeros((vec_size))   

        for l in labels:
            try:
                vec += model.get_vector(l)
            except:
                vec = vec
        vecs.append(vec)
        
    # Calculate the cosine similarity between the 2 vectors
    try: 
        cossim = np.dot(vecs[0], vecs[1])/(norm(vecs[0])*norm(vecs[1]))
    except ZeroDivisionError:
        cossim = 0
    if math.isnan(cossim):
        cossim = 0
    return cossim

def compare(label, targets, func=normalizedLevenshtein, model=None, vec_size=300, cutoff=0.2, N=1):
    """
    Compares label with a list of labels based on a given function.
    |
    label (Str):        Label
    targets (List):     List of target labels
    func (Function):    Comparison function
    cutoff (Flt):       Number that determines at what point two labels are no longer a relatedMatch but a noMatch
    |
    output (Tuple(Str, List(Tuple(Str, Str)))): (predicate, [(predicted_label, label_source)])
    """
    sources = [i[1] for i in targets]
    targets = [i[0] for i in targets]
    # Determine scores
    if model:
        scores = [func(processLabel(label),processLabel(x).lower(),model, vec_size) for x in targets]
    else:
        scores = [func(processLabel(label),processLabel(x).lower()) for x in targets]

    # Find N labels with the highest scores   
    indeces = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    m = max(scores)
    #indeces = [i for i,x in enumerate(scores) if x == m]
    
    matches = [(targets[i],sources[i]) for i in indeces]
    # If highest similarity is below cutoff return that there is no match
    if m  < cutoff:
        return ('noMatch', [('no match', 'noMatch')])
    if m == 1:
        return ('exactMatch', matches)
    else:
        return ('relatedMatch', matches)


