import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

'''
we investigate the performance of the synthetic generation model using linear regression

we convert the labels (style column) into numeric representations using the sklearn 'Label Encoder'

we see in this case, the linear regression when compared with the original data seems to think the Gaussian Copula model is more effective in synthetic generation

we realize this is sub-optimal so this examination is followed by a classification model
'''

initial_file = "video-game-dataset.csv"
synthetic_one = "video-game-dataset-extended.csv"
synthetic_two = "video-game-dataset-extended-ctgan.csv"

df_initial = pd.read_csv(initial_file)
df_syn_one = pd.read_csv(synthetic_one)
df_syn_two = pd.read_csv(synthetic_two)

# prepare labels for linear format
label_encoder = LabelEncoder()
df_initial['style'] = label_encoder.fit_transform(df_initial['style'])
df_syn_one['style'] = label_encoder.fit_transform(df_syn_one['style'])
df_syn_two['style'] = label_encoder.fit_transform(df_syn_two['style'])

# ground truth dataset
X_ground_truth = df_initial.iloc[:, :-1]
y_ground_truth = df_initial.iloc[:, -1]

# synthetic datasets
X_synthetic_1 = df_syn_one.iloc[:, :-1]
y_synthetic_1 = df_syn_one.iloc[:, -1]

X_synthetic_2 = df_syn_two.iloc[:, :-1]
y_synthetic_2 = df_syn_two.iloc[:, -1]

# train regression model on ground truth dataset
model = LinearRegression()
model.fit(X_ground_truth, y_ground_truth)

# evaluate on synthetic datasets
y_pred_1 = model.predict(X_synthetic_1)
y_pred_2 = model.predict(X_synthetic_2)

# calculate evaluation metrics
mae_1 = mean_absolute_error(y_synthetic_1, y_pred_1)
mse_1 = mean_squared_error(y_synthetic_1, y_pred_1)
r2_1 = r2_score(y_synthetic_1, y_pred_1)

mae_2 = mean_absolute_error(y_synthetic_2, y_pred_2)
mse_2 = mean_squared_error(y_synthetic_2, y_pred_2)
r2_2 = r2_score(y_synthetic_2, y_pred_2)

print(f"Synthetic Dataset 1 - MAE: {mae_1}, MSE: {mse_1}, R²: {r2_1}")
print(f"Synthetic Dataset 2 - MAE: {mae_2}, MSE: {mse_2}, R²: {r2_2}")


