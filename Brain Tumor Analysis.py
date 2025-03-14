#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load dataset

df = pd.read_csv("C:/Users/Hp/Documents/DATA ANALYSIS PROJECTS/brain_tumor_dataset.csv")


# In[3]:


#display first 5 rows
df.head()


# In[4]:


#Check data types in the dataset
print(df.info())


# In[5]:


# Summary statistics
print(df.describe())


# In[12]:


# pLotting tumor type distribution

plt.figure(figsize = (6,4))
sns.countplot(data=df, x="Tumor_Type", palette="deep")
plt.title("Distribution of Tumor Types")
plt.show()


# In[8]:


#plotting tumor size by type

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Tumor_Type", y="Tumor_Size", palette="deep")
plt.title("Tumor Size Distribution by Type")
plt.show()


# In[15]:


#select only numeric columns
numeric_df = df.select_dtypes(include=['number'])
#plotting correlation heatmap

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[16]:


#select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# check if numeric columns contain valid data
print(numeric_df.head())


# In[17]:


#plotting correlation heatmap for only numeric data
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[19]:


# Machine Learning(Predictive analysis)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[20]:


df.head()


# In[21]:


# Convert Categorical columns to numerical using Label Encoding
encoder = LabelEncoder()

#Encoding Categorical Columns

df["Gender"] = encoder.fit_transform(df["Gender"])
df["Tumor_Type"] = encoder.fit_transform(df["Tumor_Type"])
df["Location"] = encoder.fit_transform(df["Location"])
df["Histology"] = encoder.fit_transform(df["Histology"])
df["Symptom_1"] = encoder.fit_transform(df["Symptom_1"])
df["Symptom_2"] = encoder.fit_transform(df["Symptom_2"])
df["Symptom_3"] = encoder.fit_transform(df["Symptom_3"])
df["Radiation_Treatment"] = encoder.fit_transform(df["Radiation_Treatment"])
df["Surgery_Performed"] = encoder.fit_transform(df["Surgery_Performed"])
df["Chemotherapy"] = encoder.fit_transform(df["Chemotherapy"])
df["Family_History"] = encoder.fit_transform(df["Family_History"])
df["MRI_Result"] = encoder.fit_transform(df["MRI_Result"])
df["Follow_Up_Required"] = encoder.fit_transform(df["Follow_Up_Required"])


# In[22]:


#Selecting Features
X = df.drop(columns=["Patient_ID", "Tumor_Type"])

#Selecting Target Variables
y = df["Tumor_Type"]


# In[23]:


# Splitting data into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# In[24]:


#Training model
#Selecting features and target

features = ['Age', 'Tumor_Size', 'Stage', 'Tumor_Growth_Rate', 'Family_History']
target = 'Tumor_Type' #Malignant or Benign


# In[28]:


#converting Tumor_Type and Family_History data from texts to numbers

label_encoder = LabelEncoder()
df.target = label_encoder.fit_transform(df[target])
df['Tumor_Type'] = label_encoder.fit_transform(df['Tumor_Type']) #Benign=0/Malignant=1
df['Family_History'] = label_encoder.fit_transform(df['Family_History']) #Yes=1/No=0


# In[29]:


# Splitting Data into training and Testing sets

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


print(df.dtypes)


# In[ ]:


# Encoding categorical columns
categorical_cols = ['Gender', 'Stage', 'Family_History']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# In[35]:


scaler = StandardScaler()

# Select only numeric columns for scaling
numeric_cols = ['Tumor_Size', 'Tumor_Growth_Rate']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# In[34]:


print(df.head())


# In[ ]:




