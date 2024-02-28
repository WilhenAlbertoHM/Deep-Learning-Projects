import pandas as pd
from keras import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Load data
dataset = pd.read_csv("life_expectancy.csv")
print(dataset.head())
print(dataset.describe())

# Drop the Country column
dataset.drop(columns=["Country"], inplace=True)

# Get labels and features
features = dataset.drop(columns=["Life expectancy"])
labels = dataset[["Life expectancy"]]

# Convert categorical columns into numerical using one-hot-encoding
features = pd.get_dummies(dataset)

# Split into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize numerical features using ColumnTransformer
numerical_features = features.select_dtypes(include=["float64", "int64"])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder="passthrough")

# Fit transform features_train and transform features_test
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

# Build model
model = Sequential()
input = InputLayer(input_shape=(features.shape[1],))
model.add(input)
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=1))
model.summary()

# Compile model
opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss="mse", metrics=["mae"])

# Train model
model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)

# Evaluate model
res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose=0)
print(f"Final loss (RMSE): {res_mse}")
print(f"Final metric (MAE): {res_mae}")

# Save model
my_model.save("model")
