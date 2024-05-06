import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # for handling dataframes
apple_quality_data = pd.read_csv('apple_quality.csv')# Load data into dataframe


print(apple_quality_data.info())# Print information about the dataframe
print(apple_quality_data.columns[apple_quality_data.isna().any()])# Check for columns with missing values (NaN) and print those columns
print("checking for shape before dropping NaN values", apple_quality_data.shape)# Print shape of the dataframe before dropping NaN values

apple_quality_data = apple_quality_data.dropna()# Drop NaN values
print("checking for shape after dropping NaN values", apple_quality_data.shape)# Print shape of the dataframe after dropping NaN values
print(apple_quality_data.columns[apple_quality_data.isna().any()])# Check for NaN values again after dropping NaN values

apple_quality_data = apple_quality_data.drop('A_id', axis=1)# Drop unnecessary column 'A_id'
apple_quality_data.columns = ['size', 'weight', 'sweetness', 'crunchiness', 'juiciness', 'ripeness', 'acidity', 'quality']# Rename columns for clarity
apple_quality_data['acidity'] = apple_quality_data['acidity'].astype('float64')# Change data type of 'acidity' column to float64
# apple_quality_data['quality'] = apple_quality_data['quality'].map({'good':1,'bad':0})# mapping text labels to numbers

print(apple_quality_data.info())# Print information about the dataframe

#FEATURE SELECTION
apple_quality_data = apple_quality_data.sample(frac=1)

x = apple_quality_data.iloc[:,:-1].values
y = apple_quality_data.iloc[:,-1].values
print('[info] data segregated into features and target successfully...')

#DATA SPLITTING
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=45,test_size=0.3)
print('[info] data splitted into training and testing partitions successfully...')

#FEATURE ENGINEERING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)
print('[info] data scaling completed successfully...')
print('[info] data pre-processing complete...')

#MODEL TRAINING
from sklearn.linear_model import LogisticRegression#importing algo
lr = LogisticRegression(max_iter=1500)#creating an instance of algo
lr.fit(x_train_scaled,y_train)#training the model on our data
print('[info] model training complete...')

#MODEL EVALUATION
lr_pred = lr.predict(x_test_scaled)#using the trained model to predict output for x_test_scaled

from sklearn.metrics import classification_report
print("APPLE QUALITY ASSESSMENT CLASSIFICATION REPORT : \n",classification_report(y_test,lr_pred))#comparing model output with actual output

from  sklearn.metrics import confusion_matrix
lr_cf = confusion_matrix(y_test,lr_pred)
sns.heatmap(lr_cf,cmap='Blues',annot=True,fmt='d')
plt.xlabel('predicted output')
plt.ylabel('actual output')
plt.title("APPLE QUALITY ASSESSMENT CONFUSION MATRIX")
plt.savefig('apple_quality_confusion_matrix.png',bbox_inches='tight')
plt.show()

#SAVING THE TRAINED MODEL
import joblib
joblib.dump(lr, 'apple_quality_model.pkl')
print('[info] training model saved to disk...')

joblib.dump(sc, 'scaler.pkl')
print('[info] scaler saved to disk...')