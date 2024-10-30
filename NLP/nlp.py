import tensorflow as tf
import keras
import pandas
import sklearn
import pandas as pd
df = pd.read_csv('house.csv')
print(df.head())
dataset = df.values
X = dataset[:,0:10]
Y = dataset[:,10]
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
print(X_scale)
#Spliting the data for training, testing and validation
from sklearn.model_selection import train_test_split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale,
Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test,
Y_val_and_test, test_size=0.5)
#Trains the neural network with Keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential([
Dense(32, activation='relu', input_shape=(10,)), Dense(32,
activation='relu'),
Dense(1, activation='sigmoid'),
])
model.compile(optimizer='sgd', loss='binary_crossentropy',
metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=32, epochs=100,
validation_data=(X_val, Y_val))
#Evaluate the model
print("\nAccuracy:",model.evaluate(X_test, Y_test)[1])
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

'''
output

mlm@mlm-ThinkCentre-E73:~/Desktop/ROJIN$ python3 nlp.py
2024-10-30 14:33:03.337505: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-10-30 14:33:03.340588: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-10-30 14:33:03.349751: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1730278983.366045    6293 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1730278983.370796    6293 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-10-30 14:33:03.387565: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
   LotArea  OverallQual  OverallCond  ...  Fireplaces  GarageArea  AboveMedianPrice
0     8450            7            5  ...           0         548                 1
1     9600            6            8  ...           1         460                 1
2    11250            7            5  ...           1         608                 1
3     9550            7            5  ...           1         642                 0
4    14260            8            5  ...           1         836                 1

[5 rows x 11 columns]
[[0.0334198  0.66666667 0.5        ... 0.5        0.         0.3864598 ]
 [0.03879502 0.55555556 0.875      ... 0.33333333 0.33333333 0.32440056]
 [0.04650728 0.66666667 0.5        ... 0.33333333 0.33333333 0.42877292]
 ...
 [0.03618687 0.66666667 1.         ... 0.58333333 0.66666667 0.17771509]
 [0.03934189 0.44444444 0.625      ... 0.25       0.         0.16925247]
 [0.04037019 0.44444444 0.625      ... 0.33333333 0.         0.19464034]]
/home/mlm/.local/lib/python3.10/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
2024-10-30 14:33:06.234310: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
Epoch 1/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4998 - loss: 0.6632 - val_accuracy: 0.5342 - val_loss: 0.6511
Epoch 2/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.5299 - loss: 0.6502 - val_accuracy: 0.5525 - val_loss: 0.6447
Epoch 3/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.5357 - loss: 0.6513 - val_accuracy: 0.6347 - val_loss: 0.6381
Epoch 4/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6328 - loss: 0.6414 - val_accuracy: 0.6849 - val_loss: 0.6321
Epoch 5/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6735 - loss: 0.6410 - val_accuracy: 0.7215 - val_loss: 0.6265
Epoch 6/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7178 - loss: 0.6286 - val_accuracy: 0.7215 - val_loss: 0.6210
Epoch 7/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7472 - loss: 0.6165 - val_accuracy: 0.7306 - val_loss: 0.6153
Epoch 8/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7331 - loss: 0.6162 - val_accuracy: 0.7397 - val_loss: 0.6094
Epoch 9/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7397 - loss: 0.6119 - val_accuracy: 0.7397 - val_loss: 0.6035
Epoch 10/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7554 - loss: 0.6058 - val_accuracy: 0.7443 - val_loss: 0.5975
Epoch 11/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7698 - loss: 0.5978 - val_accuracy: 0.7534 - val_loss: 0.5909
Epoch 12/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7707 - loss: 0.5860 - val_accuracy: 0.7626 - val_loss: 0.5844
Epoch 13/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7952 - loss: 0.5827 - val_accuracy: 0.7900 - val_loss: 0.5783
Epoch 14/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7935 - loss: 0.5787 - val_accuracy: 0.7900 - val_loss: 0.5711
Epoch 15/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8128 - loss: 0.5678 - val_accuracy: 0.8037 - val_loss: 0.5639
Epoch 16/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8115 - loss: 0.5580 - val_accuracy: 0.8265 - val_loss: 0.5572
Epoch 17/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8372 - loss: 0.5558 - val_accuracy: 0.8128 - val_loss: 0.5494
Epoch 18/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8392 - loss: 0.5427 - val_accuracy: 0.8356 - val_loss: 0.5422
Epoch 19/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8330 - loss: 0.5382 - val_accuracy: 0.8402 - val_loss: 0.5344
Epoch 20/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8597 - loss: 0.5209 - val_accuracy: 0.8539 - val_loss: 0.5271
Epoch 21/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8536 - loss: 0.5194 - val_accuracy: 0.8539 - val_loss: 0.5194
Epoch 22/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8800 - loss: 0.5050 - val_accuracy: 0.8539 - val_loss: 0.5114
Epoch 23/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8493 - loss: 0.5145 - val_accuracy: 0.8539 - val_loss: 0.5037
Epoch 24/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8505 - loss: 0.5063 - val_accuracy: 0.8493 - val_loss: 0.4964
Epoch 25/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8550 - loss: 0.4962 - val_accuracy: 0.8493 - val_loss: 0.4887
Epoch 26/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8576 - loss: 0.4907 - val_accuracy: 0.8539 - val_loss: 0.4810
Epoch 27/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8440 - loss: 0.4814 - val_accuracy: 0.8539 - val_loss: 0.4737
Epoch 28/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8788 - loss: 0.4549 - val_accuracy: 0.8447 - val_loss: 0.4667
Epoch 29/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8451 - loss: 0.4722 - val_accuracy: 0.8539 - val_loss: 0.4594
Epoch 30/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8724 - loss: 0.4459 - val_accuracy: 0.8447 - val_loss: 0.4528
Epoch 31/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8864 - loss: 0.4288 - val_accuracy: 0.8447 - val_loss: 0.4459
Epoch 32/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8657 - loss: 0.4382 - val_accuracy: 0.8402 - val_loss: 0.4398
Epoch 33/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8753 - loss: 0.4167 - val_accuracy: 0.8402 - val_loss: 0.4334
Epoch 34/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8535 - loss: 0.4260 - val_accuracy: 0.8539 - val_loss: 0.4267
Epoch 35/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8586 - loss: 0.4304 - val_accuracy: 0.8584 - val_loss: 0.4212
Epoch 36/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8628 - loss: 0.4094 - val_accuracy: 0.8584 - val_loss: 0.4157
Epoch 37/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8768 - loss: 0.4083 - val_accuracy: 0.8630 - val_loss: 0.4098
Epoch 38/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8906 - loss: 0.4024 - val_accuracy: 0.8630 - val_loss: 0.4045
Epoch 39/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8761 - loss: 0.3938 - val_accuracy: 0.8630 - val_loss: 0.3991
Epoch 40/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8731 - loss: 0.3802 - val_accuracy: 0.8676 - val_loss: 0.3939
Epoch 41/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8641 - loss: 0.3932 - val_accuracy: 0.8676 - val_loss: 0.3891
Epoch 42/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8768 - loss: 0.3747 - val_accuracy: 0.8676 - val_loss: 0.3846
Epoch 43/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8783 - loss: 0.3629 - val_accuracy: 0.8676 - val_loss: 0.3803
Epoch 44/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8823 - loss: 0.3654 - val_accuracy: 0.8721 - val_loss: 0.3762
Epoch 45/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8876 - loss: 0.3515 - val_accuracy: 0.8721 - val_loss: 0.3724
Epoch 46/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8755 - loss: 0.3601 - val_accuracy: 0.8767 - val_loss: 0.3686
Epoch 47/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8751 - loss: 0.3578 - val_accuracy: 0.8721 - val_loss: 0.3648
Epoch 48/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8624 - loss: 0.3671 - val_accuracy: 0.8858 - val_loss: 0.3618
Epoch 49/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8791 - loss: 0.3394 - val_accuracy: 0.8813 - val_loss: 0.3582
Epoch 50/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8925 - loss: 0.3376 - val_accuracy: 0.8767 - val_loss: 0.3552
Epoch 51/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8713 - loss: 0.3334 - val_accuracy: 0.8858 - val_loss: 0.3522
Epoch 52/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8832 - loss: 0.3342 - val_accuracy: 0.8767 - val_loss: 0.3504
Epoch 53/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8967 - loss: 0.3112 - val_accuracy: 0.8858 - val_loss: 0.3467
Epoch 54/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8723 - loss: 0.3398 - val_accuracy: 0.8767 - val_loss: 0.3448
Epoch 55/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8713 - loss: 0.3439 - val_accuracy: 0.8813 - val_loss: 0.3421
Epoch 56/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8564 - loss: 0.3375 - val_accuracy: 0.8858 - val_loss: 0.3395
Epoch 57/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8793 - loss: 0.3487 - val_accuracy: 0.8904 - val_loss: 0.3373
Epoch 58/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8829 - loss: 0.3058 - val_accuracy: 0.8767 - val_loss: 0.3353
Epoch 59/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8806 - loss: 0.3114 - val_accuracy: 0.8813 - val_loss: 0.3332
Epoch 60/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8813 - loss: 0.3179 - val_accuracy: 0.8676 - val_loss: 0.3316
Epoch 61/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8595 - loss: 0.3269 - val_accuracy: 0.8676 - val_loss: 0.3297
Epoch 62/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8912 - loss: 0.2983 - val_accuracy: 0.8813 - val_loss: 0.3279
Epoch 63/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8822 - loss: 0.2993 - val_accuracy: 0.8721 - val_loss: 0.3262
Epoch 64/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8853 - loss: 0.2997 - val_accuracy: 0.8676 - val_loss: 0.3257
Epoch 65/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8786 - loss: 0.3159 - val_accuracy: 0.8676 - val_loss: 0.3233
Epoch 66/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8694 - loss: 0.3148 - val_accuracy: 0.8767 - val_loss: 0.3219
Epoch 67/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8921 - loss: 0.2843 - val_accuracy: 0.8767 - val_loss: 0.3206
Epoch 68/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8719 - loss: 0.3187 - val_accuracy: 0.8721 - val_loss: 0.3194
Epoch 69/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8937 - loss: 0.2812 - val_accuracy: 0.8721 - val_loss: 0.3183
Epoch 70/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8871 - loss: 0.2874 - val_accuracy: 0.8721 - val_loss: 0.3182
Epoch 71/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8846 - loss: 0.2858 - val_accuracy: 0.8767 - val_loss: 0.3157
Epoch 72/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8769 - loss: 0.2853 - val_accuracy: 0.8721 - val_loss: 0.3146
Epoch 73/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8760 - loss: 0.2938 - val_accuracy: 0.8767 - val_loss: 0.3134
Epoch 74/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8841 - loss: 0.2884 - val_accuracy: 0.8721 - val_loss: 0.3126
Epoch 75/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8782 - loss: 0.2835 - val_accuracy: 0.8721 - val_loss: 0.3126
Epoch 76/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8685 - loss: 0.3210 - val_accuracy: 0.8767 - val_loss: 0.3104
Epoch 77/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8598 - loss: 0.3318 - val_accuracy: 0.8767 - val_loss: 0.3096
Epoch 78/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8700 - loss: 0.2982 - val_accuracy: 0.8721 - val_loss: 0.3090
Epoch 79/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8711 - loss: 0.3046 - val_accuracy: 0.8767 - val_loss: 0.3077
Epoch 80/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8866 - loss: 0.2823 - val_accuracy: 0.8721 - val_loss: 0.3069
Epoch 81/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8947 - loss: 0.2768 - val_accuracy: 0.8767 - val_loss: 0.3061
Epoch 82/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8787 - loss: 0.2907 - val_accuracy: 0.8721 - val_loss: 0.3070
Epoch 83/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8794 - loss: 0.2800 - val_accuracy: 0.8721 - val_loss: 0.3050
Epoch 84/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8850 - loss: 0.2754 - val_accuracy: 0.8721 - val_loss: 0.3038
Epoch 85/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8783 - loss: 0.2893 - val_accuracy: 0.8721 - val_loss: 0.3042
Epoch 86/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8931 - loss: 0.2764 - val_accuracy: 0.8721 - val_loss: 0.3025
Epoch 87/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8921 - loss: 0.2637 - val_accuracy: 0.8721 - val_loss: 0.3017
Epoch 88/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8876 - loss: 0.2914 - val_accuracy: 0.8721 - val_loss: 0.3013
Epoch 89/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8877 - loss: 0.2878 - val_accuracy: 0.8721 - val_loss: 0.3017
Epoch 90/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8858 - loss: 0.2933 - val_accuracy: 0.8721 - val_loss: 0.2998
Epoch 91/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8837 - loss: 0.2901 - val_accuracy: 0.8721 - val_loss: 0.2993
Epoch 92/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9000 - loss: 0.2706 - val_accuracy: 0.8721 - val_loss: 0.2989
Epoch 93/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8718 - loss: 0.2958 - val_accuracy: 0.8721 - val_loss: 0.2993
Epoch 94/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8949 - loss: 0.2654 - val_accuracy: 0.8721 - val_loss: 0.2980
Epoch 95/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9000 - loss: 0.2696 - val_accuracy: 0.8721 - val_loss: 0.2971
Epoch 96/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8692 - loss: 0.3172 - val_accuracy: 0.8676 - val_loss: 0.2984
Epoch 97/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8973 - loss: 0.2539 - val_accuracy: 0.8721 - val_loss: 0.2964
Epoch 98/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8864 - loss: 0.2634 - val_accuracy: 0.8721 - val_loss: 0.2956
Epoch 99/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9000 - loss: 0.2443 - val_accuracy: 0.8721 - val_loss: 0.2950
Epoch 100/100
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8852 - loss: 0.2815 - val_accuracy: 0.8721 - val_loss: 0.2946
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8890 - loss: 0.3122  

Accuracy: 0.8949771523475647
'''
