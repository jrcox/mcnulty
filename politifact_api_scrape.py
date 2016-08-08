#importing names from politifact api
import requests
import pandas as pd
import time

people_url =  'http://www.politifact.com/api/people/all/json/'
response = requests.get(url=people_url)
df = pd.read_json(response.text)
df['full_name'] = df['first_name'] + ' ' + df['last_name']
df['party_'] = df.party.apply(lambda x: x['party'])

#generating list of all parties
party_list = list(set(df['party_']))

#selecting for people within 3 main parties
parties = ['Democrat', 'Republican', 'Independent']
df = df[df['party_'].isin(parties)]

#making list of politifact names
politifact_names = list(df['full_name'])

#making list of nameslugs for api requests
name_slug_list = list(set(df['name_slug']))

#making API request for all people who are categorized as Democrat, Republican or Ind.
the_url = 'http://www.politifact.com/api/statements/truth-o-meter/people/{}/json/?n=100'
df_list = list()
for name in name_slug_list:
        try:
            full_url = the_url.format(name)
            response = requests.get(full_url)
            df_list.append(pd.read_json(response.text))
            time.sleep(3)
        except:
            pass

df_statements = pd.concat(df_list).reset_index(drop=True)


