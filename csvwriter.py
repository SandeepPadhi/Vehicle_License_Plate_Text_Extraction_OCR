
# importing pandas module 
import pandas as pd 

''' 
# reading csv file 
df= pd.read_csv("trainValOriginal.csv") 
df = df[['track_id','image_path','lp','train','Output']]# displying  dataframe - Output 1 
#print(df.describe()) 
df['Output'][5]="SandeepPadhi"
df['Output'][6]="SandeepPadhi"

print(df['Output'][5])

df.to_csv('Outval2.csv')
# inserting column with static value in data frame 
#data.insert(2, "Outputval", "Any") 
#print(df.info())
  
# displaying data frame again - Output 2 
#data.head() 

'''
df= pd.read_csv("trainValOriginal.csv") 
df = df[['track_id','image_path','lp','train','Output']]# displying  dataframe - Output 1 


pathAdd=df['image_path']

for i in range(0,len(pathAdd)):
    path=str(df['image_path'][i])
    print(type(path))
    if 'm' not in path:
        df.drop(df.index[[i]])

df.to_csv('Outval2.csv')



