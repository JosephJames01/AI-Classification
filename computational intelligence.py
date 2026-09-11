import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv('C:/Users/joeha/OneDrive/Documents/PythonScripts/ai/garments_worker_productivity.csv',
                 parse_dates=['date'])
df['wip'] = df['wip'].fillna(0)
df['date'] = df['date'].dt.dayofyear
df['quarter'] = df['quarter'].map({'Quarter1': 1, 'Quarter2': 2, 'Quarter3': 3, 'Quarter4': 4,
                                   'Quarter5': 5})
df['day'] = df['day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5,
                            'Saturday': 6, 'Sunday': 7})
df['department'] = df['department'].map({'sewing': 1, 'finishing': 2,
                                         'finishing ': 2})

numerical_features = ['date', 'quarter', 'team', 'smv', 'wip', 'over_time',
                       'incentive', 'idle_time', 'idle_men', 'no_of_style_change',
                       'no_of_workers', 'department']

# Binary classification target
df['actual_productivity'] = (df['actual_productivity'] >= 0.75).astype(int)

# ---------------------------------------------------------------
# Train / validation / test split 
# 70% train, 15% validation, 15% test
# ---------------------------------------------------------------
np.random.seed(42)
n = len(df)
shuffled_idx = np.random.permutation(n)

train_frac, val_frac = 0.70, 0.15
train_end = int(n * train_frac)
val_end = int(n * (train_frac + val_frac))

train_idx = shuffled_idx[:train_end]
val_idx = shuffled_idx[train_end:val_end]
test_idx = shuffled_idx[val_end:]

df_train = df.iloc[train_idx].copy()
df_val = df.iloc[val_idx].copy()
df_test = df.iloc[test_idx].copy()

# Normalise using TRAIN min/max only, then apply the same transform to val/test.
# Fitting min/max on the full dataset would leak test-set information into training.
train_min = df_train[numerical_features].min()
train_max = df_train[numerical_features].max()
denom = (train_max - train_min).replace(0, 1)  # guard against divide-by-zero on constant columns

for split_df in (df_train, df_val, df_test):
    split_df[numerical_features] = (split_df[numerical_features] - train_min) / denom

X_train = df_train[numerical_features].values
y_train = df_train['actual_productivity'].values.reshape(-1, 1)
X_val = df_val[numerical_features].values
y_val = df_val['actual_productivity'].values.reshape(-1, 1)
X_test = df_test[numerical_features].values
y_test = df_test['actual_productivity'].values.reshape(-1, 1)

print(f"Train: {X_train.shape[0]} rows | Val: {X_val.shape[0]} rows | Test: {X_test.shape[0]} rows")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Neural network structure
input_neurons = X_train.shape[1]
hidden_neurons = 20
output_neurons = 1

weights_0 = 2 * np.random.rand(input_neurons, hidden_neurons) - 1
weights_1 = 2 * np.random.rand(hidden_neurons, output_neurons) - 1

learning_rate = 0.002
epochs = 20000 

train_loss_history = []
val_loss_history = []


def forward(X):
    hidden_output = sigmoid(np.dot(X, weights_0))
    output = sigmoid(np.dot(hidden_output, weights_1))
    return hidden_output, output


for epoch in range(epochs):
    # Forward propagation — TRAIN data only
    hidden_layer_output, predicted_output = forward(X_train)
    error = y_train - predicted_output

    if epoch % 1000 == 0:
        train_loss = np.mean(np.abs(error))
        _, val_pred = forward(X_val)
        val_loss = np.mean(np.abs(y_val - val_pred))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_1.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating weights
    weights_1 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_0 += X_train.T.dot(d_hidden_layer) * learning_rate


# Final evaluation  TEST set only

_, test_pred = forward(X_test)

tp = np.sum((test_pred >= 0.75) & (y_test >= 0.75))
fp = np.sum((test_pred >= 0.75) & (y_test < 0.75))
fn = np.sum((test_pred < 0.75) & (y_test >= 0.75))
tn = np.sum((test_pred < 0.75) & (y_test < 0.75))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
accuracy = (tp + tn) / (tp + fp + fn + tn)
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print('\n--- Test set performance (held-out, unseen during training) ---')
print(f'f1_score: {f1_score:.4f}')
print(f'precision: {precision:.4f}')
print(f'recall: {recall:.4f}')
print(f'accuracy: {accuracy:.4f}')

# Plotting train vs validation loss over epochs
plt.plot([x * 1000 for x in range(len(train_loss_history))], train_loss_history, label='Train loss')
plt.plot([x * 1000 for x in range(len(val_loss_history))], val_loss_history, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs. Validation Loss over Epochs')
plt.legend()
plt.show()
