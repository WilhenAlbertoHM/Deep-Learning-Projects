import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Display all columns
pd.set_option("display.max_columns", None)

"""
Predict whether a student will be admitted to a graduate program
This is a regression task, a probability between 0 and 1.
"""

# Read and explore data
df = pd.read_csv("admissions_data.csv")
print(df.head())
print(df.describe())
print(df.columns)
print(df["University Rating"].value_counts())

# Make sure columns don't have extra spaces
df.columns = df.columns.str.strip()

# Split data into features and labels
df.drop("Serial No.", axis=1, inplace=True)
features = df.loc[:, df.columns != "Chance of Admit"]
labels = df.loc[:, df.columns == "Chance of Admit"]

# Split data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Build model
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(features_train_scaled.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

# Compile and train model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(features_train_scaled, labels_train, 
                    epochs=100, batch_size=1, verbose=1, 
                    validation_split=0.2, 
                    callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)])

# Evaluate model
loss, mae = model.evaluate(features_test_scaled, labels_test)
print("Loss: ", loss)
print("Mean Absolute Error: ", mae)

# Make predictions
predictions = model.predict(features_test_scaled)

# Calculate R^2 score
r2 = r2_score(labels_test, predictions)
print("R2 score: ", r2)

# Plot MAE and validation MAE per epoch
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history["mae"])
ax1.plot(history.history["val_mae"])
ax1.set_title("Model MAE")
ax1.set_ylabel("MAE")
ax1.set_xlabel("# of Epochs")
ax1.legend(["Train", "Validation"], loc="upper left")

# Plot MSE and validation MSE per epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history["loss"])
ax2.plot(history.history["val_loss"])
ax2.set_title("Model MSE")
ax2.set_ylabel("Loss (MSE)")
ax2.set_xlabel("# of Epochs")
ax2.legend(["Train", "Validation"], loc="upper left")

# Keep plots from overlapping each other  
fig.tight_layout()
plt.savefig("graduate_admissions_regressor.png")
plt.show()

# Save model
model.save("model")