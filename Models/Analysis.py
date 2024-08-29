import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from dataReader import getLabels

def intersection(list1, list2):
    """
    Returns the intersection of the 2 lists
    """
    return [x for x in list1 if x in list2]

def union(list1, list2):
    """
    Returns the union of the 2 lists
    """
    return list1 + list2

def removeList(list1, list2):
    """
    Removes all elements from list1 that also occur in list2
    """
    for x in list2:
        try:
            list1.remove(x)
        except ValueError:
            pass
    return list1

def relativeComplements(lists):
    """
    Find relative complement for each model
    """

    result = []
    
    for list in lists:
        Ns = [intersection(list,x) for x in lists if x != list]
        
        relative_complement = list.copy()

        for N in Ns:
            relative_complement = removeList(relative_complement,N)
        result.append(relative_complement)

    return result

def getFiles(path):
    files = []

    for filename in os.listdir("./Analysis"):
        if filename[:len(path)] == path:
            files.append(filename)

    return files

def plot(df, files, path='predictions_GPT'):
    data_count= [(len(df[df['Model'] == x]),x[len(path)-3:-4]) if x in set(list(df['Model'])) else (0,x[len(path)-3:-4]) for x in files]
    dfcount = pd.DataFrame(data=data_count, columns=['#Label', 'Model'])

    fig, ax = plt.subplots()

    ax.bar(dfcount['Model'], dfcount['#Label'])
    plt.xticks(rotation=45,ha='right')

    plt.show()


def getCorrectPredictions(files, filter=['TP', 'TN']):
    correct_per_file = []
    res_files = []

    for file in files:
        correct = []

        f = open(f"./Analysis/{file}", "r", encoding="utf-8")
        lines = f.readlines()

        lines = [x for x in lines if x[:2] in filter]

        for l in lines:
            try:
                double = [x.strip() for x in re.search('\(.*\)', l).group(0)[1:-1].split("@")[1:]]
            except:
                double = ''
        
            if double:
                correct.append((double[0],double[1],double[2]))

        if correct:
            correct_per_file.append(correct)
            res_files.append(file)

    return(res_files, correct_per_file)


if __name__ == "__main__":
    path = 'predictions_path_GPT'
    files = getFiles(path)
    filter = ['TP', 'TN', 'FP', 'FN']

    files,correct_per_file = getCorrectPredictions(files)
    pilots,labels,_,_,_,_,_,_ = getLabels()

    pilotdict = {}
    for key, value in zip(labels, pilots):
        value = value.lower().split(':')[0]
        key = key.strip().lower()
        if key in pilotdict:
            if value not in pilotdict[key]:
                pilotdict[key] += [value]
        else:
            pilotdict[key] = [value]

    data = []
    complements = relativeComplements(correct_per_file)
    for f,c in zip(files,complements):
        for x in c:
            try:
                p = ' '.join(pilotdict[x[0]])
            except:
                p = 'Error'
            data.append((p,x[0],x[1],x[2],f))

    df = pd.DataFrame(data=data, columns=['Pilot', 'Label', 'Match', 'Ontology', 'Model'])
    #df.to_excel(excel_writer="./Analysis/test.xlsx", index=False)

    data2 = []
    for f,c in zip(files,correct_per_file):
        for x in c:
            try:
                p = ' '.join(pilotdict[x[0]])
            except:
                p = 'Error'
            data2.append((p,x[0],x[1],x[2], f))

    df2 = pd.DataFrame(data=data2, columns=['Pilots', 'Label', 'Match', 'Ontology', 'Model'])
    df2.to_excel(excel_writer="./Analysis/test.xlsx", index=False)




    #plot(df, files, path)

    