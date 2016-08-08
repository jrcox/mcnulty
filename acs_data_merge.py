import pandas as pd
import pickle

#importing docs from American community survey 2007-2014
table_years = range(10,15)
table_years_2 = range(7,10)
list_of_files = ['~/downloads/ACS_{}_1YR_S0501_with_ann_edit.csv'.format(i) for i in table_years]
list_of_files_2 = ['~/downloads/ACS_0{}_1YR_S0501_with_ann_edit.csv'.format(i) for i in table_years_2]
state_files = list_of_files + list_of_files_2

#converting to dataframes
df_07_110_state = pd.read_csv(state_files[5], index_col = 'Id', header = 1)
df_08_110_state = pd.read_csv(state_files[6], index_col = 'Id', header = 1)
df_09_111_state = pd.read_csv(state_files[7], index_col = 'Id', header = 1)
df_10_111_state = pd.read_csv(state_files[0], index_col = 'Id', header = 1)
df_11_112_state = pd.read_csv(state_files[1], index_col = 'Id', header = 1)
df_12_113_state = pd.read_csv(state_files[2], index_col = 'Id', header = 1)
df_13_113_state = pd.read_csv(state_files[3], index_col = 'Id', header = 1)
df_14_114_state = pd.read_csv(state_files[4], index_col = 'Id', header = 1)

#cleaning data
databases = [df_07_110_state, df_08_110_state, df_09_111_state, df_10_111_state, df_11_112_state, df_12_113_state,
            df_13_113_state, df_14_114_state]

for db in databases:
    db.rename(columns = {'Geography':'state'}, inplace = True)
    
df_07_110_state['year'] = 2007
df_08_110_state['year'] = 2008
df_09_111_state['year'] = 2009
df_10_111_state['year'] = 2010
df_11_112_state['year'] = 2011
df_12_113_state['year'] = 2012
df_13_113_state['year'] = 2013
df_14_114_state['year'] = 2014


for db in databases:
    db['year'] = db['year'].apply(str)

def new_id(y):
    return db['state'] + db['year']

    
for db in databases:
    db['state_year'] = db['state'] + db['year']

for db in databases:
    db.index = db['state_year']
    
#creating state dataframe
master_state_dataframe=pd.concat(databases)

#opening politifact frame
with open("senate_politifact.pkl", 'rb') as senatepolipkl: 
    senate_total = pickle.load(senatepolipkl)
with open('governor_politifact.pkl', 'rb') as govpolipkl:
    gov_total = pickle.load(govpolipkl)

#masterdf = master_state_dataframe.merge(senate_master, on = ['state', 'year'])
masterdf_senate = pd.merge(master_state_dataframe, senate_total, on=['state', 'year'])
masterdf_governor = pd.merge(master_state_dataframe, gov_total, on=['state', 'year'])
masterdf = masterdf_senate.append(masterdf_governor)

#pickling dataframe
with open('masterdf.pkl', 'wb') as masterdfpkl:
    pickle.dump(masterdf, masterdfpkl)


    