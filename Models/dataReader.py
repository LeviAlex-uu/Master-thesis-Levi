import os
import re
import numpy as np
import pandas as pd

from rdflib import Graph

def getLabels(path=os.path.dirname(__file__) + '\mapping.xlsx', num_sheets=22):
    """
    Retrieves labels from the manual mapping file
    |
    path (str):           Location of mapping.xlsx file
    num_sheets (int):     Determines the number of sheets from the file are read (only used for testing)
    """
    xls = pd.ExcelFile(path)

    sheet_names = ['P1_gearbox', 'P1_pitch_system', 'P1_generator', 'P3_ELCE', 'P3_measurements', 'P3_TO_TTP', 'P3_TO_CEV', 
                'P4_inputs', 'P4_outputs', 'P4_look_ahead', 'P4_ipto', 'P4_generation', 'P4_desfa', 'P4_entsog', 
                'P5_ASM', 'P5_PMU', 'P5_EV', 'P6', 'P7_EF_comp', 'P7_Energy_Efficiency', 'P7_Solar_panels', 'P7_Solar_comp']

    pilot = []
    subject_labels = []     # Data label
    object_id = []          # Path in the ontology        
    object_labels = []      # Ontology label    
    predicate = []          # Relation
    object_source = []      # Object URI
    object_ontology = []    # Object ontology
    context = []            # Additional information on the subject label

    for sheet_name in sheet_names[:num_sheets]:
        sheet = pd.read_excel(xls, sheet_name)

        pilot = np.concatenate([pilot, sheet['subject_id'].to_numpy()])
        subject_labels = np.concatenate([subject_labels, sheet['subject_label'].to_numpy()])
        object_id = np.concatenate([object_id, sheet['object_id'].to_numpy()])
        object_labels = np.concatenate([object_labels, sheet['object_label'].to_numpy()])   
        predicate = np.concatenate([predicate, sheet['predicate_id'].to_numpy()])
        object_source = np.concatenate([object_source, sheet['object_source'].to_numpy()])
        object_ontology = np.concatenate([object_ontology, sheet['object_ontology'].to_numpy()])
        context = np.concatenate([context, sheet['comment'].to_numpy()])    
    
    return (pilot,subject_labels,object_id,object_labels,predicate,object_source,object_ontology,context)
    
# Process subject labels

def processLabel(label):
    """
    Process label by removing delimiters
    |
    label (str):    Label to be processed
    """
    # Remove delimiters
    label = label.replace('_', ' ')
    label = ' '.join(re.split(r'\W', label))

    # Split on uppercase letters
    label = re.findall('.[^A-Z]*',label)

    # Recombine & Lowercase label
    label = ' '.join([x.strip() for x in label]).lower()

    return label

def getOntologies(path=os.getcwd()+'.\Ontologies'):    
    """
    Retrieves labels from the ontology files
    |
    path (Str):     Location of folder containing the ontologies
    """  
    labels = []

    q = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?label
    WHERE {
        ?p rdfs:label ?label
        FILTER (lang(?label) IN ("en"))
    }
    """
   
    for filename in os.listdir(path) :
        g = Graph()
        g.parse(path+ '.\\' + filename)

        query_res = g.query(q)

        for row in query_res:
            label = ' '.join([l.strip().lower() for l in row['label'].split()])
            labels.append((label, filename))

        #print(f'{filename}, number of labels: {len(query_res)}')

    return(labels)
