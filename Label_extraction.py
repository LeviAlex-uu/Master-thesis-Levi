import os
import xml.dom.minidom as xdm
import pandas as pd
import numpy as np
import glob

def read_files(path='C:\\Users\\vanheesla\\Documents\\VeeAlign-master\\datasets\\conference\\alignments'):
    domains = []
    for filename in os.listdir(path):
        dom = xdm.parse(path + '\\' + filename)
        domains.append(dom)
    return domains

def get_pairs(doms):
    result = []

    for dom in doms:
        entities1 = dom.getElementsByTagName('entity1')
        entities2 = dom.getElementsByTagName('entity2')

        pairs = [(x.getAttribute('rdf:resource').replace('http://', ''), y.getAttribute('rdf:resource').replace('http://', '')) for x,y in zip(entities1,entities2)]
        result += pairs

    return result


def find_file_type(type='xlsx',path='C:\\Users\\vanheesla\\OneDrive - TNO\\Attachments\\Thesis\\Enershare\\Teamwork\\Teamwork5\\dataset-extracts\\Consumption'):
    type =  '*.' + type

    fpaths = glob.glob(path+os.sep+type)
    fnames = [fp.split(os.sep)[-1] for fp in fpaths]
    return fpaths

def read_xlsx():
    headers = []

    for f in find_file_type(type='xlsx'):
        df = pd.read_excel(f,nrows=1)
        print(df)

def read_mappings(path=''):
    xls = pd.ExcelFile('mapping.xlsx')
    dt = pd.read_excel(xls, 'P1_gearbox')
    print(dt)

    
def main():
    #doms = read_files()
    #pairs = get_pairs(doms)
    #print(pairs)
    #print(len(pairs))
    read_mappings()

if __name__ == '__main__':
    main()




