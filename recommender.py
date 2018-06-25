
# coding: utf-8

# In[1]:


import pandas as pd
pd.options.display.max_columns = None
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


data = pd.read_csv('dataset/frappe/frappe.csv', sep="\t")
appData = pd.read_csv('dataset/frappe/meta.csv', sep="\t")


# In[3]:


data.head()


# In[4]:


data = data.merge(appData, on='item')
#removing not used columns
data = data.drop('package', axis=1)
data = data.drop('icon', axis=1)


# ### Merge both files

# In[5]:


data.head()


# ### check the shape of the DataFrame (rows, columns)

# In[6]:


data.shape


# ### Most used apps

# In[7]:


data.name.value_counts()[:5]


# In[8]:


feature_cols = ['cnt', 'downloads']


# In[9]:


#All rows and the feature columns
X = data.loc[:, feature_cols]


# In[10]:


X.shape


# In[11]:


y = data.rating


# In[12]:


y.shape


# In[13]:


data.head()


# ### Converts columns that are strings into numbers and store the labels in the labels array
# ### the index of the labels array corresponds to the column number

# In[14]:


labels = [None] * 20
for x in range(0, 20):
    if type(data.iloc[:, x][0]) == str:
        data.iloc[:, x], labels[x] = pd.factorize(data.iloc[:, x])
    else:
        labels[x] = None


# In[15]:
#Separa a coluna de item para ser o target
y = data['item']
data = data.drop('item', axis=1)

#Aplica os n vizinhos para treinar a base
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data.loc[:], y)


## COLUNAS
#user cnt daytime weekday isweekend homework cost wheater country city category downloads developer language description name price rating shortdesc
print(knn.predict([[5000, 0, 2, 1, 0, 5, 0, 1, 2, 1, 0, 3, 2, 1, 1, 0, 1, 0, 1]]))
#print(data)

