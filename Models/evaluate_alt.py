import os
import re
import numpy as np
import baselines as bs
import dataReader as dr

from tqdm import tqdm
from gensim.models import KeyedVectors

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def evaluate_model(cutoff_range=[0.2], num_of_returns=1, print_results=False):
    """
    Evaluate model, evaluates all models from baselines.py with data from the Enershare project
    """
    _,subject_labels, ids, true_labels, _, _, true_sources,_ = dr.getLabels(num_sheets=22)
    print(f'Number of subject labels: {len(subject_labels)}')

    ontology_labels = dr.getOntologies()
    print(f'Number of ontology labels: {len(ontology_labels)}')

    path_name = os.path.dirname(__file__)
    model = KeyedVectors.load(path_name + '\word2vec.d2v')

    #predicted_predicates = []
    true_positives = []
    false_positives = []
    false_negatives = []
    true_negatives = []
    mrrs = []
    halfaccs = []
    ids = [x.lower().split(':') for x in ids]
    
    for c in cutoff_range:
        predictedLe = [] 
        predictedLcs = []
        predictedW2V = []

        # Get predictions from each chosen model
        for subject in tqdm(subject_labels):
            predictedW2V.append(bs.compare(subject,ontology_labels, bs.word2Vec, model, cutoff=c, N=num_of_returns))

        predicted = [predictedLe,predictedLcs,predictedW2V]


        # For each of the baselines loop through the predictions to calculate the individual accuracies
        for predicted_labels in predicted:
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            hits = 0
            pp = []
            ranks = []

            for predictions,true_label,true_source,id in zip(predicted_labels, true_labels, true_sources, ids):
                pp.append(predictions[0])
                predicted_labels = predictions[1]
                rank = 0

                # When there are multiple predictions (same score) for one label loop through them
                for p in predicted_labels:
                    pred_label = p[0]                       # Get the predicted label
                    pred_source = re.split('\\.', p[1])     # Get the predicted source    

                    # Accuracy when comparing the predicted label with the path to the label in the ontology
                    if pred_label in id and true_source in pred_source:
                        if pred_label == 'no match':
                            tn += 1
                            break
                        else:
                            tp += 1

                    elif pred_label == 'no match' and 'noMatch' not in id:
                        fn += 1
                        break

                    else:
                        fp += 1 

                    
            fp = 574 - (tp+fn+tn)
            
            true_positives.append(tp)
            false_positives.append(fp)
            false_negatives.append(fn)
            true_negatives.append(tn)
            halfaccs.append(hits) 
            #predicted_predicates.append(pp)
            mrrs.append(ranks)     
        
    # Print results
    models = ['Levenshtein','LCS','Word2Vec']
    if print_results:
        for i,_ in enumerate(models):
            temp_accs = true_positives[i::len(models)]
            temp_hits = halfaccs[i::len(models)]
            print(f'Model: {models[i]}')
            for j,(t,h) in enumerate(zip(temp_accs,temp_hits)):
                print(f'Correct predictions: {t}/{len(subject_labels)}, Accuracy: {round(t/len(subject_labels)*100,2)}, cutoff: {cutoff_range[j]}')
                print(f'Correct predictions: {h}/{len(subject_labels)}, G-Accuracy: {round(h/len(subject_labels)*100,2)}, cutoff: {cutoff_range[j]}')
        
    return (models, (true_positives, false_positives, false_negatives, true_negatives), halfaccs, mrrs)

def castMatrix(data, dtype=int):
    return list(map(lambda sub: list(map(dtype, sub)), data))

def translation(x):
    dict = {'noMatch' : 0, 0 : 'noMatch', 'relatedMatch' : 1, 1 : 'relatedMatch', 'exactMatch' : 2, 2 : 'relatedMatch'}
    return dict[x]

if __name__ == "__main__":
    path = os.path.dirname(__file__)
    _, (tp, fp, fn, tn), halfaccs, mrrs = evaluate_model([0.7],1)
    np.savetxt(path + '.\\Results\\results_alt.txt', [tp,fp,fn,tn])
    #np.savetxt(path + '.\\Results\\results_halfaccs3.txt', halfaccs)
    #true_predicates = [translation(x) for x in predicates[0]]
    #predicates = castMatrix(predicates[1], translation)
    #np.savetxt(path + '.\\Results\\results_true_predicates5.txt', true_predicates)
    #np.savetxt(path + '.\\Results\\results_predicates5.txt', predicates)
    mrrs = [sum(x) for x in mrrs]
    #np.savetxt(path + '.\\Results\\results_mrrs.txt', mrrs)



