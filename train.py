import numpy as np
import zarr
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from optuna_integration import KerasPruningCallback
import optuna
import os

from src.cosine_annealing_warmup import CosineAnnealingWarmRestarts


X = zarr.open('/mnt/d/fire_dataset/X.zarr', mode='r')
y = zarr.open('/mnt/d/fire_dataset/y.zarr', mode='r')

if os.path.exists('indices.npz'):
    indices = np.load('indices.npz')
    idx_train = indices['idx_train']
    idx_val = indices['idx_val']
    idx_test = indices['idx_test']
else:
    idx_trainval, idx_test = train_test_split(np.arange(X.shape[0]), test_size=0.1, random_state=42)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.1, random_state=42)
    np.savez('indices.npz', idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

WD_CH = 11

def rotate_batch(X, y, k):
    """
    Rotate in 90° steps.
    k in {0,1,2,3} -> 0°,90°,180°,270°
    """
    if k == 0:
        return X, y

    X_rot = np.rot90(X, k=k, axes=(2, 3))
    y_rot = np.rot90(y, k=k, axes=(1, 2))

    # adjust wd channel after rotation
    wd = X_rot[..., WD_CH]
    wd = (wd + 0.25 * k) % 1
    X_rot[..., WD_CH] = wd

    return X_rot, y_rot


def flip_batch_horizontal(X, y):
    """
    Horizontal flip over width axis.
    """
    X_f = np.flip(X, axis=3)
    y_f = np.flip(y, axis=2)

    # adjust wd channel after horizontal flip
    wd = X_f[..., WD_CH]
    wd = 1 - wd
    X_f[..., WD_CH] = wd

    return X_f, y_f

def data_generator(indices, batch_size, randomize=False):
    N = len(indices)
    while True:
        if randomize:
            np.random.shuffle(indices)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]

            Xb = X[batch_idx].copy()
            yb = y[batch_idx].copy()

            if randomize:
                for i in range(len(Xb)):
                    k = np.random.randint(0, 4)
                    Xi, yi = Xb[i:i+1], yb[i:i+1]
                    Xi, yi = rotate_batch(Xi, yi, k)

                    if np.random.rand() < 0.5:
                        Xi, yi = flip_batch_horizontal(Xi, yi)

                    Xb[i] = Xi[0]
                    yb[i] = yi[0]

            yield Xb, yb

inp = layers.Input(shape=(None, *X.shape[2:]))

x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=False,
    activation="relu",
)(x)
x = layers.Conv2D(
    filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"
)(x)

epochs = 50
batch_size = 5

train_gen = data_generator(idx_train, batch_size, randomize=True)
val_gen = data_generator(idx_val, batch_size)

def objective(trial):
    T_0 = trial.suggest_int('T_0', 5, 25)
    T_mult = trial.suggest_int('T_mult', 1, 2)
    eta_max = trial.suggest_float('eta_max', 1e-3, 1e-1, log=True)
    eta_min = trial.suggest_float('eta_min', 1e-6, 1e-4, log=True)

    reduce_lr = CosineAnnealingWarmRestarts(T_0, T_mult, eta_max, eta_min)
    prune = KerasPruningCallback(trial, monitor='val_loss')
    checkpoint = ModelCheckpoint(f'model_{trial.number}.keras',
                                 monitor='val_loss',
                                 save_best_only=True, 
                                 mode='min',
                                 verbose=1)

    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
    )

    history = model.fit(
        train_gen,
        batch_size=batch_size,
        steps_per_epoch=len(idx_train) // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=len(idx_val) // batch_size,
        callbacks=[checkpoint, reduce_lr, prune],
    )

    return history.history['val_loss'][-1]

pruner = optuna.pruners.HyperbandPruner()

study = optuna.create_study(
    storage='sqlite:///fire-perimeter.db',
    study_name='fire-perimeter',
    direction='minimize',
    pruner=pruner,
    load_if_exists=True
)

study.optimize(objective, n_trials=16)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of failed trials: ", len(failed_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

