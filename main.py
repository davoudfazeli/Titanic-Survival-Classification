import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data_train = pd.read_csv("dataset/train.csv") # Load Train Data

# Preprocessing Train Data
data_train.replace(["male", "female"],[1,0], inplace=True)
data_train.fillna(0,inplace=True)
x_train = np.array(data_train[["Pclass", "Sex", "Age", "SibSp", "Fare", "Parch"]])
y_train = np.array(data_train["Survived"])

# Definition The Model of ML

model = tf.keras.models.Sequential([
                                   tf.keras.layers.Dense(6, "sigmoid"),     # input Layer
                                   tf.keras.layers.Dense(128, "relu"),      # First Hidden Layer
                                   tf.keras.layers.Dense(8, "sigmoid"),     # Second Hidden Layer
                                   tf.keras.layers.Dense(2, "sigmoid")])    # Output Hidden Layer
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
# Train Model
output = model.fit(x_train,y_train, epochs = 45)

# Plot Train Results

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0.4)
(ax1, ax2) = gs.subplots(sharex=True)
epl = np.arange(1, 46)
ax1.plot(epl, output.history["accuracy"])
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoches")
ax1.set_xticks(np.arange(1, 46, 4))

ax2.plot(epl, output.history["loss"])
ax2.set_title("Loss")
ax2.set_xlabel("Epoches")

# Load Test Data
data_test = pd.read_csv("dataset/test.csv")
label_test = pd.read_csv("dataset/gender_submission.csv")
# Preprocessing Test Data
data_test.replace(["male", "female"],[1,0], inplace=True)
data_test.fillna(0,inplace=True)
data_test.head(10)
x_test = np.array(data_test[["Pclass", "Sex", "Age", "SibSp", "Fare", "Parch"]])
y_test = np.array(label_test["Survived"])


model.evaluate(x_test,y_test) # Model Evaluation

# Prediction Results for two person
p = model.predict(np.array([[3,1,18,0,0,0],[1,0,18,3,70,1]]))
np.argmax(p,axis = 1)