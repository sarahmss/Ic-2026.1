import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
TS = 0.07
TIME_STEPS = 20
BATCH_SIZE = 32
EPOCHS = 20

DATA_PATH = "./02-Filtered/02-SimpleLossSavGol/02-Ts-15/X/Data/Datasets.xlsx"

TARGET_NAME = "X"
PREDICTORS = ["PwmD", "PwmE", "sPwm", "dPwm", "cosTheta"]

TITLES = ["Train_1", "Train_2", "Val_1", "Val_2", "Test_1", "Test_2", "LSG_1", "LSG_2"]

# =========================
# LOAD TREINO
# =========================
df = pd.concat([pd.read_excel(DATA_PATH, sheet_name="Train_1"),
               pd.read_excel(DATA_PATH, sheet_name="Train_2")])

Target = df[TARGET_NAME].values
dTarget = df[f"d{TARGET_NAME}"].values

pred_raw = df[PREDICTORS].values

# =========================
# NORMALIZAÇÃO
# =========================
pred_mean = pred_raw.mean(axis=0)
pred_std = pred_raw.std(axis=0) + 1e-8
pred_raw = (pred_raw - pred_mean) / pred_std

y_mean = dTarget.mean()
y_std = dTarget.std() + 1e-8
y_raw = (dTarget - y_mean) / y_std

# =========================
# SLIDING WINDOW
# =========================
def create_sequences(pred, target, time_steps):
    Xs, ys = [], []
    for i in range(len(pred) - time_steps):
        Xs.append(pred[i:i+time_steps])
        ys.append(target[i+time_steps])
    return np.array(Xs), np.array(ys)

X, y = create_sequences(pred_raw, y_raw, TIME_STEPS)

# =========================
# MODELO (dinâmico)
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(TIME_STEPS, len(PREDICTORS))),
    tf.keras.layers.SimpleRNN(32, activation='tanh'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mse'
)

# =========================
# TREINO
# =========================
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

# =========================
# FIGURA GERAL
# =========================
n = len(TITLES)
fig, axs = plt.subplots(n, 2, figsize=(14, 3*n))

for i, title in enumerate(TITLES):

    df = pd.read_excel(DATA_PATH, sheet_name=title)

    Target = df[TARGET_NAME].values
    dTarget = df[f"d{TARGET_NAME}"].values

    pred_raw = df[PREDICTORS].values

    # normalização treino
    pred_raw = (pred_raw - pred_mean) / pred_std
    y_raw = (dTarget - y_mean) / y_std

    X_test, y_test = create_sequences(pred_raw, y_raw, TIME_STEPS)

    # predição
    y_pred = model.predict(X_test, verbose=0)

    y_pred = y_pred * y_std + y_mean
    y_true = y_test * y_std + y_mean

    # =========================
    # RECONSTRUÇÃO (se for derivada)
    # =========================
    Target_true = Target[TIME_STEPS:]

    Target_rec = np.zeros_like(y_pred.flatten())
    Target_rec[0] = Target_true[0]

    for k in range(1, len(Target_rec)):
        Target_rec[k] = Target_rec[k-1] + y_pred[k] * TS

    # =========================
    # PLOT TARGET
    # =========================
    axs[i, 0].plot(Target_true, label="Real", alpha=0.7)
    axs[i, 0].plot(Target_rec, label="Reconstruído", linestyle='--')

    axs[i, 0].set_title(f"{title} - {TARGET_NAME}")
    axs[i, 0].grid(True)

    if i == 0:
        axs[i, 0].legend()

    # =========================
    # PLOT DERIVADA
    # =========================
    axs[i, 1].plot(y_true, label=f"d{TARGET_NAME} Real", alpha=0.7)
    axs[i, 1].plot(y_pred, label=f"d{TARGET_NAME} Predito", linestyle='--')

    axs[i, 1].set_title(f"{title} - d{TARGET_NAME}")
    axs[i, 1].grid(True)

    if i == 0:
        axs[i, 1].legend()

axs[-1, 0].set_xlabel("Amostra")
axs[-1, 1].set_xlabel("Amostra")

plt.suptitle(f"Generalização - {TARGET_NAME}", fontsize=16)
plt.tight_layout()
plt.show()