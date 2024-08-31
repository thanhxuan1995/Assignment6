import os
from os.path import join, isdir, isfile
import glob
import datetime
import time
from datetime import datetime, timedelta
import pandas as pd

def read_file(product,date_back, work_dic, purpose):
    ##create filename
    now = datetime.now()
    file_name = []
    for x in range(date_back):
        d = now - timedelta(days = x)
        day = d.strftime("%Y-%m-%d")
        file_name.append(day)
    ## read file from os.
    open_file =[]
    dic = join(work_dic,purpose,product)
    # Check whether the specified path exists or not
    if os.path.exists(dic):
        files = os.listdir(dic)
        for i in range(len(files)):
            item = files[i][:10]
            if item in file_name:
                open_file.append(files[i])

    dff= pd.DataFrame()
    for file in open_file:
        file_dic = join(work_dic,purpose,product,file)
        df = pd.read_csv(file_dic, low_memory= False)
        print(len(df))
        dff = pd.concat([dff,df], axis= 0, ignore_index=True)
        dff = dff.drop_duplicates()
        print(file_dic)
    return dff

## update LC if we working at finish
def LC_update(dff, work_dic, product):
    domain = 'Loss_Code_Kitchen'
    LC_file = os.listdir(join(work_dic, domain))
    for i in LC_file:
        if (product in i) & ('csv' in i):
            lc = pd.read_csv(join(work_dic, domain,i), low_memory = False)
    update_lc = lc[['filter_vids', 'ws_loss_code']]
    dfff = pd.merge(left = dff,right = update_lc, on= 'filter_vids', how = 'left')
    dffe = pd.concat([dfff,lc], axis= 0, ignore_index=True)
    dffef = dffe.drop_duplicates()
    dffef['ws_loss_code'] = dffef['ws_loss_code'].fillna('good')
    return dffef

