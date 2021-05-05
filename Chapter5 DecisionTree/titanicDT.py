import pandas as pd
import numpy as np
from CART import *


#%% Processing the train.csv  Dividing them into training set and validation set
df = pd.read_csv('../titanic/train.csv')
df.fillna(method='ffill',inplace=True)
df.drop(['Name','Ticket','Cabin'],inplace=True,axis=1)
df['Sex'].replace('female',0,inplace=True)
df['Sex'].replace('male',1,inplace=True)
for i,val in enumerate(['S','C','Q']):
    df['Embarked'].replace(val,i,inplace=True)
df = df.sample(frac=1)
# 80% of the dataset will be used to train and the rest of the to validation!
train_rate = 0.8
divide = int(df.shape[0] * train_rate)
full_data = df.iloc[:,2:].to_numpy()
full_label = df.iloc[:,1].to_numpy()
train_data = full_data[:divide]
train_label = full_label[:divide]
validation_data = full_data[divide:]
validation_label = full_label[divide:]

#%% Construct the decision tree and test the result on validation set
dt = DecisionTree(classifier=True,criterion='gini',min_sample=20)
dt.fit(train_data,train_label)
prediction = dt.predict(validation_data)
right = np.sum(prediction == validation_label)
all = len(prediction)
print("Train Accuracy:{}/{}  {:.2f}%".format(right,all,right/all * 100))
alpha_subtrees = dt.cost_loss_pruning_path(train_data,train_label)
for alpha,root in alpha_subtrees:
    # print(alpha)
    prediction = dt.predict(validation_data,root=root)
    right = np.sum(prediction == validation_label)
    all = len(prediction)
    print("Train Accuracy when alpha = {:.6f}:{}/{}  {:.2f}%".format(alpha,right, all, right / all * 100))


df_test = pd.read_csv('../titanic/test.csv')
df_test.fillna(method='ffill',inplace=True)
df_test.drop(['Name','Ticket','Cabin'],inplace=True,axis=1)
df_test['Sex'].replace('female',0,inplace=True)
df_test['Sex'].replace('male',1,inplace=True)
for i,val in enumerate(['S','C','Q']):
    df_test['Embarked'].replace(val,i,inplace=True)
#

test_data = df_test.iloc[:,1:].to_numpy()
prediction = dt.predict(train_data,root=alpha_subtrees[10][-1])
right = np.sum(prediction == train_label)
all = len(prediction)
print("Train Accuracy:{}/{}  {:.2f}%".format(right,all,right/all * 100))
# print("Decision Tree Depth:",dt.depth)
test_prediction = dt.predict(test_data,root=alpha_subtrees[30][-1])
print("Survived/all : {}/{}".format(np.sum(test_prediction),len(test_prediction)))
test_result = np.column_stack([df_test['PassengerId'].values,test_prediction])
test_result = pd.DataFrame(data=test_result,columns=['PassengerId','Survived'])
test_result.to_csv('../titanic/result.csv',index=False)


# train_data = df.iloc[:,2:].to_numpy()
# train_label = df.iloc[:,1].to_numpy()
# # data.plotDataset(test_data)
# dt = DecisionTree(classifier=True,criterion='gini',min_sample=20)
# dt.fit(train_data,train_label)
#



# indexes = pd.notna(df['Age']).to_numpy().nonzero()[0]
# df.fillna(method='ffill',inplace=True)



