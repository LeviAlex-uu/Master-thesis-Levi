import os
import re
import sys
import random
import codecs

import numpy as np
import matplotlib as plt

from tqdm import tqdm
from dataReader import getLabels, getOntologies, processLabel
from baselines import normalizedLevenshtein

def labelDict(data):
    dictionary = {}
    for key,val1,val2 in data:
        key = key.strip().lower()
        if key in dictionary:
            if (val1.lower(),val2.lower() + '.txt') not in dictionary[key]:
                dictionary[key] += [(val1.lower(), val2.lower() + '.txt')]
        else:
            dictionary[key] = [(val1.lower(), val2.lower() + '.txt')]
    return dictionary

def labelDict2(data):
    dictionary = {}
    for key,val1,val2,val3 in data:
        key = key.strip().lower()
        if key in dictionary:
            if (val1.lower(),val2.lower(), val3.lower() + '.txt') not in dictionary[key]:
                dictionary[key] += [(val1.lower(),val2.lower(), val3.lower() + '.txt')]
        else:
            dictionary[key] = [(val1.lower(),val2.lower(), val3.lower() + '.txt')]
    return dictionary

def evaluatePredictions(predictions, dictionary, path=False, write=False, output_file='./Analysis/test.txt'):
    analysis=[]
    tp,tn,fp,fn = 0,0,0,0

    for i,(l,m,s) in enumerate(predictions):
        l = l.lower()
        m = m.lower()
        s = s.lower().replace('.txt','') + '.txt'

        try:
            if not path:                            # Only check if intitial prediction matches correct concept from mapping.xlsx
                tuples = [(x,z) for (x,_,z) in dictionary[l]]
                if (m, s) in tuples:                    # First check if (match, source) tuple exists in the dictionary[label] -> check if prediction is correct
                    if m in ['nomatch', 'no match']:        # Check if true negative
                        tn += 1
                        analysis.append(f"TN: ({i}@ {l}@ {m}@ {s})")
                    else:                                   # Else true positive
                        tp += 1
                        analysis.append(f"TP: ({i}@ {l}@ {m}@ {s})") 
                
                elif m in ['nomatch', 'no match']:      # Check if prediction is false negative
                    if (m,s) not in tuples:
                        fn += 1
                        analysis.append(f"FN: ({i}@ {l}@ {m}@ {s})")
                else:                                   # Else prediction is a false positive
                    fp += 1
                    analysis.append(f"FP: ({i}@ {l}@ {m}@ {s})")
                        
            else:                                   # Also check if the prediction is on the path to the concept from mapping.xlsx
                tuples = [((y+f':{x}').split(':'),z) for (x,y,z) in dictionary[l.lower()]]
                if any([(m in x) and (s == y) for x,y in tuples]):
                    if m in ['nomatch', 'no match']:        # Check if true negative
                        tn += 1
                        analysis.append(f"TN: ({i}@ {l}@ {m}@ {s})")
                    else:                                   # Else true positive
                        tp += 1
                        analysis.append(f"TP: ({i}@ {l}@ {m}@ {s})") 
                    
                elif m in ['nomatch', 'no match']:      # Check if prediction is false negative
                    if not any([(m == x) and (s == y) for x,y in tuples]):
                        fn += 1
                        analysis.append(f"FN: ({i}@ {l}@ {m}@ {s})")
                else:                                   # Else prediction is a false positive
                    fp += 1
                    analysis.append(f"FP: ({i}@ {l}@ {m}@ {s})")
        except:
            pass #print(l)
        
    print(tp + tn + fp + fn)
    print(tp, tn, fp, fn)

    N = len(predictions)
    try:
        accuracy = round((tp+tn)/N,3)
    except:
        accuracy = 0
    try:    
        precision = round(tp/(tp+fp),3)
    except:
        precision = 0    
    try:    
        recall = round(tp/(tp+fn),3)
    except:
        recall = 0


    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}\n')

    if write:
        f = open(output_file, 'w', encoding='utf-8')
        f.write('\n'.join(analysis))
        f.close()


def labelDictAlt(data):
    dictionary = {}
    for key,val1,_ in data:
        key = key.strip().lower()
        if key in dictionary:
            if val1 not in dictionary[key]:
                dictionary[key] += [val1.lower().replace(' ', '')]
        else:
            dictionary[key] = [val1.lower().replace(' ', '')]
    return dictionary

def evaluate_predictions_alt(predictions, dictionary, path = False, write=False):
    correct = []
    tp, fp, tn, fn = 0, 0, 0, 0
    for i,(l,m,s) in enumerate(predictions):
        if l[0] == '"':
            l = l[1:-1]
        try:
            if not path:
                if m.replace(' ', '') in dictionary[l]:
                    tp += 1
                    correct.append(f"{i}, {l}, {m}")
                elif m == 'nomatch':
                    if m in dictionary[l]:
                        tn += 1
                    elif m not in dictionary[l]:
                        fn += 1
                        correct.append(f"{i}, {l}, {m}")
                else:
                    fp += 1
            else:
                for path in dictionary[l]: 
                    if m.replace(' ', '') in path.split(':'):
                        tp += 1
                        correct.append(f"{i}, {l}, {m}")
                        break
                    elif m == 'nomatch':
                        if m in path.split(':'):
                            tn += 1
                            break
                        elif m.replace(' ', '') not in path.split(':'):
                            fn += 1
                            correct.append(f"{i}, {l}, {m}")
                            break
                    else:
                        fp += 1
                        break
        except:
            pass #print(l)
    t = sum([1 for _,x,_ in predictions if x == 'nomatch'])
    print(tp, fp, tn, fn)
    print(f'Labels not found in manual mapping: {len(predictions)-(tp+fp+tn+fn)}')
    print(f'noMatch count: {t}')

    N = len(predictions)
    try:
        accuracy = round((tp+tn)/N,3)
    except:
        accuracy = 0
    
    precision = round(tp/(tp+fp),3)
    recall = round(tp/(tp+fn),3)

    if write:
        f = open('./Analysis/CorrectLabelsFT_sv.txt', 'w', encoding='utf-8')
        f.write('\n'.join(correct))
        f.close()

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}\n')

def findClosestLabel(label,source):
    
    labels = getOntologies()
    targets = [x for x,y in labels]# if y.lower() == source+'.ttl']

    if targets == []:
        return label

    scores = [normalizedLevenshtein(x,label) for x in targets]
    m = max(scores)
    indeces = [i for i,x in enumerate(scores) if x == m]
    matches = [targets[i] for i in indeces]

    if m == 1:
        return label
    else:
        return processLabel(random.choice(matches))
    
if __name__ == "__main__":
    #print(normalizedLevenshtein('BarometricPressureProperty'.lower(), 'calorific value property'.lower()))
    #print(normalizedLevenshtein('BarometricPressureProperty'.lower(), 'PressureProperty'.lower()))
    #print(findClosestLabel('BarometricPressureProperty'.lower(),'EnershareProperty'.lower()))
    #exit()

    _,true_labels,true_paths,true_matches,_,true_sources,true_ontologies,_ = getLabels(path='mapping.xlsx')
    dictalt = labelDictAlt(zip(true_labels,true_matches,true_ontologies))
    dictalt2 = labelDictAlt(zip(true_labels,true_paths,true_ontologies))
    dictionary = labelDict2(zip(true_labels,true_matches,true_paths,true_ontologies))

    for filename in os.listdir("./Results"):
        if filename[:6] != 'output':
            continue

        f = open(f"./Results/{filename}", "r", encoding="utf-8")
        lines = [x for x in f.readlines() if x != '\n']

        pred_labels = []
        pred_matches = []
        pred_sources = []
        for l in lines:
            if re.search('{.*}', l) != None:
                try:
                    pred_labels.append(re.search('Label:(.*)Match:', l).group(1)[1:-2].strip().lower())
                    pred_matches.append(re.search('Match:(.*)Ontology:', l).group(1)[1:-2].lower())

                    source = re.search('Ontology:(.*)Score:', l).group(1)[1:-2].lower().replace('.txt', '')
                    if 'nomatch' in source.split():
                        pred_sources.append('nomatch')
                    else:
                        pred_sources.append(source)
                except Warning:
                    print('Something went wrong')

        predictions = list(zip(pred_labels,pred_matches,pred_sources))

        print(filename[6:])
        evaluatePredictions(predictions, dictionary, write=False, output_file=f"./Analysis/predictions_{filename[6:]}")
        evaluatePredictions(predictions, dictionary, True, write=True, output_file=f"./Analysis/predictions_path_{filename[6:]}")

    exit()
    t = []

    for (l,m,s) in tqdm(predictions):
        n = findClosestLabel(m,s)
        t.append((l,n,s))

    t = '\n'.join([f'{{Label: {x}, Match: {y}, Ontology: {z}, Score: 0}}' for x,y,z in t])

    try:
        s = codecs.decode(codecs.encode(t, 'utf-8', 'replace'), 'utf-8')
    except:
        s = None

    f = open('./Results/Modified_fewshot_ft2.txt', 'w', encoding='utf-8')
    f.write(s)
    f.close()