# --------------------------------------------------------------------------- #
# MIT license
# Copyright 2020, Southeast University, Liu Pengxiang. All Rights Reserved.
# 
# File: Reader.py
# Version: 1.0.0
# --------------------------------------------------------------------------- #
'''
A tool for reading the data from GasLib.
'''

import os
import sys
import tqdm
import openpyxl
import zipfile
import xmltodict
import numpy as np


# tool for excel files
class excel_tool(object):

    def __init__(self):
        pass

    # save excel
    def save_as_excel(self, filename, excel_dict):
        work_book = openpyxl.Workbook()
        for sheet in excel_dict:
            if len(excel_dict[sheet]['value']) > 0:
                work_sheet = work_book.create_sheet()
                work_sheet.title = sheet
                work_sheet.append(excel_dict[sheet]['title'])
                for row in tqdm.tqdm(excel_dict[sheet]['value']):
                    work_sheet.append(row)
        work_book.remove(work_book['Sheet'])
        work_book.save(filename = os.path.join(save, filename))


# get the data of network
class get_data_of_network(excel_tool):

    def __init__(self, net_files, name):
        super(get_data_of_network, self).__init__()
        data = self.read_data(net_files[0])
        self.save_as_excel(name.replace('.zip', '-network.xlsx'), data)
    
    def read_data(self, name):
        with zfile.open(name, mode = 'r') as fd:
            doc = xmltodict.parse(fd.read())
        excel_dict = {}
        framwork_id = ['nodes', 'connections']
        for _id in framwork_id:
            pool = doc['network']['framework:' + _id]
            for item_type in pool:
                excel_dict[item_type] = []
                info = {'title': [], 'value': []}
                if not isinstance(pool[item_type], list):
                    pool[item_type] = [pool[item_type]]
                for i, item in enumerate(pool[item_type]):
                    title, value = self.get_attributes(item)
                    if i == 0:
                        info['title'] = title
                        info['value'].append(value)
                    else:
                        info['value'].append(value)
                # get date
                excel_dict[item_type] = info
        return excel_dict
    
    def get_attributes(self, item):
        #
        title, value = [], []
        for attr in item:
            if isinstance(item[attr], str):
                title.append(attr.replace('@', ''))
                value.append(item[attr])
            if isinstance(item[attr], dict):
                if len(item[attr]) == 2:
                    title.append(attr + '(' + item[attr]['@unit'] + ')')
                else:
                    title.append(attr)
                value.append(item[attr]['@value'])
        return title, value


# get the data of nomination
class get_data_of_nomination(excel_tool):

    def __init__(self, scn_files, name):
        super(get_data_of_nomination, self).__init__()
        data = self.read_data(scn_files)
        self.save_as_excel(name.replace('.zip', '-nomination.xlsx'), data)
    
    def read_data(self, scn_files):
        excel_dict = {'flow': [], 'pressure': []}
        for key in excel_dict.keys():
            title, index, value, i = ['id'], [], [], 0
            for i, scn_file in enumerate(tqdm.tqdm(scn_files)):
                with zfile.open(scn_file, mode = 'r') as fd:
                    doc = xmltodict.parse(fd.read())
                    value.append([])
                scenario = doc['boundaryValue']['scenario']
                title.append(scenario['@id'])
                for item in scenario['node']:
                    if i == 0:
                        index.append(item['@id'])
                    if key in item.keys():
                        node_value = self.get_attributes(item, key)
                        value[i].append(node_value)
            if key in item.keys():
                value.insert(0, index)
            else:
                value = []
            value = np.array(value, dtype = object).T.tolist()
            excel_dict[key] = {'title': title, 'value': value}
        return excel_dict
    
    def get_attributes(self, item, key):
        if isinstance(item[key], dict):
            if item['@type'] == 'exit':
                value = item[key]['@value']
            if item['@type'] == 'entry':
                value = '-' + item[key]['@value']
            return '[' + value + ',' + value + ']'
        if isinstance(item[key], list):
            if item['@type'] == 'exit':
                lower = item[key][0]['@value']
                upper = item[key][1]['@value']
            if item['@type'] == 'entry':
                lower = '-' + item[key][0]['@value']
                upper = '-' + item[key][1]['@value']
            return '[' + lower + ',' + upper + ']'



# main function
if __name__ == "__main__":
    # 
    root = os.getcwd()
    # name = 'GasLib-134-v2.zip'
    name = input("Please input the name of .zip file: ")
    save = name.replace('.zip', '')
    try:
        zipfile_path = os.path.join(root, name)
        with zipfile.ZipFile(zipfile_path, mode = 'r') as zfile:
            if not os.path.exists(save):
                os.mkdir(save)
            # determine the type of nomination
            files = zfile.namelist()
            nominations = [i for i in files if i[-1] == '/']
            if len(nominations) == 0:
                scn_fold = ''
            if len(nominations) == 1:
                scn_fold = nominations[0]
            if len(nominations) >= 1:
                print('The following types of nominations are found:')
                for i, nomi in enumerate(nominations):
                    print('({}) {}'.format(i + 1, nomi.replace('/','')))
                n = input("Please enter the type number (1/2): ")
                scn_fold = nominations[int(n) - 1]
            print('Start reading:')
            # .net files
            string = '.net'
            net_files = [i for i in files if string in i]
            get_data_of_network(net_files, name)
            # .scn files
            string = '.scn'
            scn_files = [i for i in files if string in i and scn_fold in i]
            get_data_of_nomination(scn_files, name)
        print('Finished!')
    except:
        print('Somthing wrong!')