#fuzzy text matching senators, governors and politifact API statements

#opening politifact dataset
with open("df_politifact.pkl", 'rb') as politifactpickl: 
    df = pickle.load(politifactpickl)
    
#importing senator roster from everypolitician.org
list_of_files = ['~/downloads/term-{}_1.csv'.format(i) for i in range(108,115)]

list_ = []
for file_ in list_of_files:
    dfs = pd.read_csv(file_,index_col=None, header=0)
    list_.append(dfs)
senateframe = pd.concat(list_)

columns = ['name', 'group', 'area', 'chamber', 'term', 'start_date', 'end_date', 'gender']
senateframe = senateframe[columns]

#governors database import from csv
governors = pd.read_csv('~/desktop/governors.csv', header = 0)

#text matching between three datasets
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

senate_names_ = senateframe['name']
governor_names_ = governors['name']
politifact_names = df['name']

#senate name matching
sen_new = [process.extract(x, politifact_names, limit=3) for x in senate_names_]
lab = ""
i = 1
while i <= 3:
    lab = lab + " " + "Match" + str(i)
    i += 1

sen_matches = pd.DataFrame(sen_new, columns = lab.split())
 
ds={}
for x in range(1,4):
    ds["Match{0}".format(x)]=[y[0] for y in sen_matches['Match'+str(x)]]
 
    ds['using_original'] = senate_names_
 
    #match1 = [x[0] for x in fhp_matches['Match1']]
    ds['perfect_match'] = ds['Match1'] == ds['using_original']
ds = pd.DataFrame(ds)

#governor name matching
gov_new = [process.extract(x, politifact_names, limit=3) for x in governor_names_]
lab = ""
i = 1
while i <= 3:
    lab = lab + " " + "Match" + str(i)
    i += 1

gov_matches = pd.DataFrame(gov_new, columns = lab.split())
 
dg={}
for x in range(1,4):
    dg["Match{0}".format(x)]=[y[0] for y in gov_matches['Match'+str(x)]]
 
    dg['using_original'] = governor_names_
 
    #match1 = [x[0] for x in fhp_matches['Match1']]
    dg['perfect_match'] = dg['Match1'] == dg['using_original']
dg = pd.DataFrame(dg)

#function to calculate matching ratio
def fuzzratio(row):
    try: 
        return fuzz.ratio(row['name'], row['Match1'])
    except:
        return 0.
senate_match['ratio'] = senate_match.apply(fuzzratio, axis = 1)
gov_match['ratio'] = gov_match.apply(fuzzratio, axis = 1)

#function to replace name with politifact name if the match ratio was greater than .9
def name_replace(row):
    if row['ratio'] >= 90:
        return row['Match1']
    else:
        return row['name']
        
#create senate and governor name matched dataframes
senate_match['ratio'] = senate_match.apply(fuzzratio, axis = 1) #create ratio column by applying function
gov_match['ratio'] = gov_match.apply(fuzzratio, axis = 1)

#merge with statements dataframe 
senate_politifact = senate_match.merge(politifact_df, on = ['name', 'congress'])
governors_politifact = governors_match.merge(politifact_df, on = ['name', 'year'])
