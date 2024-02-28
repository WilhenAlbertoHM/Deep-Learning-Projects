import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from keras.utils import to_categorical

# Load data
data = pd.read_csv("heart_failure.csv")

# Print data
print(data.head())
print(data.describe())
print(data.info())

# Print distribution of death_event column
print(Counter(data["DEATH_EVENT"]))

# Split data into features and labels; x and y, respectively
x = data[["age", "anaemia", "creatinine_phosphokinase", "diabetes", 
                 "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine", 
                 "serum_sodium", "sex", "smoking", "time"]]
y = data["DEATH_EVENT"]

# Convert categorical features from x to one-hot encoding vectors
x = pd.get_dummies(x)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale numeric features in dataset
ct = ColumnTransformer([("numeric", StandardScaler(), ["age", "creatinine_phosphokinase", 
                                                        "ejection_fraction", "platelets", "serum_creatinine", 
                                                        "serum_sodium", "time"])])

# Scale X_train and X_test
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Label encode Y_train and Y_test
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

# Turn Y_test and Y_train into binary vectors
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Create and compile model
model = Sequential()
model.add(InputLayer(X_train.shape[1],))
model.add(Dense(12, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)

# Evaluate model
loss, acc = model.evaluate(X_test, Y_test)

# Print loss and accuracy
print("Loss:", loss)
print("Accuracy:", acc)

# Make predictions
y_estimate = model.predict(X_test)
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(Y_test, axis=1)

# Print classification report
report = classification_report(y_true, y_estimate)
print(report)

# Save model
model.save("model")

