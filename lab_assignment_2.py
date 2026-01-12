import numpy as np
import pandas as pd
file_name="Lab Session Data.xlsx"
df=pd.read_excel(file_name)
print('data:',df)
X=df[['Candies (#)',"Mangoes (Kg)",'Milk Packets (#)']]
Y=df['Payment (Rs)']
print('Matrix_x',X)
print('Matrix_Y',Y)
rank = np.linalg.matrix_rank(X)
print("Rank of the feature matrix X =", rank)
X_pinv = np.linalg.pinv(X)
cost_vector = X_pinv.dot(Y)

print("\nEstimated Cost of Each Product:")
print("Candies  (Rs per piece):", cost_vector[0])
print("Mangoes (Rs per kg):", cost_vector[1])
print("Milk Packets (Rs per packet):", cost_vector[2])
count=1
for i in Y:
    if i>200:
        print(f'c{count} is rich')
    elif i<200:
        print(f'c{count} is poor')
    count+=1
    