import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import pydicom

sns.set(color_codes=True)

custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']

df = pd.read_csv("D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\train.csv")

#Exploratory Data Analysis
df.info()

df.isnull().sum()

print("Q. How many patients are present in train set ?")
print("A.",df['Patient'].nunique())
print("------------")
print("Q. How many unique ages are present in train set ?")
print("A.",df['Age'].nunique())
print("------------")
print("Q. How many smoking statuses are present in train set ?")
print("A.",df['SmokingStatus'].nunique())

#How 'FVC', 'Percent' and 'Age' are distributed
fig, axes = plt.subplots(1, 3, figsize=(17,5))
fig.suptitle('Distribution of different features')

sns.distplot(df['FVC'], color='blue', ax=axes[0])
sns.distplot(df['Percent'], color='orange', ax=axes[1])
sns.distplot(df['Age'],color='green', ax=axes[2])

plt.show()

print ("No of Patient : " , df.Patient.unique().shape)

df_gb = df.groupby(['Patient','Age','Sex','SmokingStatus']).count()

print("Max record of patient : " , df_gb["Weeks"].max())

print("Min record of patient : " , df_gb["Weeks"].min())

df_gb.hist()

plt.show()

df_gb2 = df_gb.reset_index()

print('Age distribution')
sns.distplot(df_gb2['Age'],bins=35)

print('Sex distribution')
print(df_gb2['Sex'].value_counts())

print('SmokingStatus distribution')
print(df_gb2['SmokingStatus'].value_counts())

b = df_gb2[['Age','Sex']]
flg = b['Sex']=='Male'
b['Sex'] = flg
b['Sex']=b['Sex'].astype(int)

print("Relation between Age and Sex")
sns.jointplot(x="Age", y="Sex", data=b, kind="kde")
plt.show()

patient_dir = "D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\train\\ID00038637202182690843176"

datasets = []

# First Order the files in the dataset
files = []
for dcm in list(os.listdir(patient_dir)):
    files.append(dcm) 
files.sort(key=lambda f: int(re.sub('\D', '', f)))

# Read in the Dataset
for dcm in files:
    path = patient_dir + "/" + dcm
    datasets.append(pydicom.dcmread(path))

# Plot the images
fig=plt.figure(figsize=(16, 6))
columns = 10
rows = 3

for i in range(1, columns*rows +1):
    img = datasets[i-1].pixel_array
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="gray")
    plt.title(i, fontsize = 9)
    plt.axis('off');

plt.show()
