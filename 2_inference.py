#LOADING TRAINED MODEL
import joblib
apple_model = joblib.load('apple_quality_model.pkl')
sc = joblib.load('scaler.pkl')
print('[info] trained model and scaler loaded successfully...')

import warnings
warnings.filterwarnings('ignore')

size = float(input("enter size : "))
weight = float(input("enter weight : "))
sweetness = float(input("enter sweetness : "))
crunchiness = float(input("enter crunchiness : "))
juiciness = float(input("enter juiciness : "))
ripeness = float(input("enter ripeness : "))
acidity = float(input("enter acidity : "))

new_user_input = [[size, weight, sweetness, crunchiness, juiciness, ripeness, acidity]]
new_user_input_scaled = sc.transform(new_user_input)

new_user_output = apple_model.predict(new_user_input)[0]

print("\n MODEL DIAGNOSIS : ")
print('-'*30)
print(f"according to the given parameters, the apple is {new_user_output}")




