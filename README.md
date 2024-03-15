# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
![311415388-83f4ff2c-edd2-4191-9421-bf727013ebf8](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/7948ade6-6738-41ee-9eae-d55bd258c7eb)
```
dt.info()
```
![311415429-d75d3d97-c83d-4d2b-bbf7-071485cb2187](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/bd2b97f3-6a83-4992-ac65-bb6fa72f7eb2)
```
dt.shape
```
![311415471-10f26b11-c859-4ca8-860d-765c5edf09de](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/98cfab90-22f5-4bbd-a71a-3631b826b71d)
```
dt.set_index("PassengerId",inplace=True)
dt.describe()
```
![311415503-716f82d5-d97a-4666-a113-ab39c226ee2a](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/f4bbbb33-1e8a-45ff-bef1-f69e5fcdedaf)
```
dt.nunique()
```
![311415524-99249907-8415-4dee-962f-f41d75c5158c](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/90b97cb8-c343-4d6e-a8fd-073148fb9be0)
```
dt["Survived"].value_counts()
```
![311415554-646f84ff-7fb7-43d6-87df-8bf3f4e169d1](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/79e0c6ad-9597-4584-901b-25480a918b52)
```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![311415581-c41679de-84c8-49bf-b4ad-2ab91ec9b489](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/c790f979-e9f3-470d-925b-3ad02fd79e66)
```
sns.countplot(data=dt,x="Survived")
```
![311415612-c2e65eb0-78c2-4cab-96f9-19b940c74ce8](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/1d751734-1268-47cb-b1b1-99536d4f4987)
```
dt
```
![311416478-6970fea4-0ff2-4194-bf19-55eb3c97830a](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/9e4f2ca6-b820-4a43-91b7-6d48599ad5b1)
```
dt.Pclass.unique()
```
![311415781-7eb9f3c4-151b-4973-a69d-531caccd63c1](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/197bb6f0-deee-4f04-ba81-c19f1b5c89ea)
```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![311415848-31820297-9d0c-4ad2-b426-add463c8e562](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/3facfe80-540b-4b21-8ba4-ee130f5eedac)
```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
![311415881-71535f1a-31eb-4667-bcaf-cc7d1c6f3395](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/7564e152-2a26-454d-a148-160d2592cb73)
```
sns.catplot(x='Survived',hue="Gender",data=dt,kind='count')
```
![311415910-fbf4a83a-0a70-4dc5-80cc-0fc6267a87ce](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/01ba926e-cefd-40f6-9350-b8084d069630)
```
dt.boxplot(column="Age",by="Survived")
```
![311415958-7645eb0b-dd11-4b75-bdd9-e3f422b522ef](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/c12b4b3b-23a8-4dc7-b236-5952165ba6c2)
```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![311416010-cdbb0911-3aad-40f5-a84d-0bd22de66b66](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/f1231627-5984-4ba4-8051-07cafc4bf141)
```
sns.jointplot(x="Age",y="Fare",data=dt)
```
![311416041-338e74f4-f66c-4896-a6de-8cc0446337cf](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/7035c334-74a0-47af-bed4-30932d82003d)
```
fig,ax1=plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax1,x="Pclass",y="Age",hue="Gender",data=dt)
```
![311416095-0d2340f5-38c8-4242-b241-4e7c959f3b61](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/1c18a69d-7db5-4a80-b3ff-9a2e5b9aa3f0)
```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![311416151-3cfc8650-d94d-4e80-80b7-c913b9ac4bdb](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/9f5942bb-c6a3-4736-a91f-1aa13bb3fd63)
```
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![311416190-15d8888c-2ec2-47ec-b8b0-322bce02dd74](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/1f0bedbc-c445-400c-ae85-1d122c59c1a6)
```
sns.pairplot(dt)
```
![311416251-ea0e6e4b-1a73-458b-8cbb-a83071374afe](https://github.com/sujigunasekar/EXNO2DS/assets/119559822/8383884e-a5d8-4af8-8f96-77316074a623)





# RESULT
Thus, the Exploratory Data Analysis on the given data set was performed successfully.
