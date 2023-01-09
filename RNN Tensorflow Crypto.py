import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM, BatchNormalization #CuDNNLSTM
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LEN = 60 #use last 60 minutes as context
FUTURE_PERIOD_PREDICT = 3 #1 period = 1 minute
RATIO_TO_PREDICT = "BTC-USD"

#hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):
        return 1
    else: 
        return 0

def preprocess_df(df):
    df = df.drop('future', 1) #remove future column

    for col in df.columns:
        if col != "target": #don't need to scale target
            df[col] = df[col].pct_change()
            df.dropna(inplace=True) #remove N/A entries
            df[col] = preprocessing.scale(df[col].values) #scale all values from 0-1
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN) #auto-pops out old items as new ones are inserted
    # deque of lists, where each list represents 1 time 

    # for c in df.columns:
    #     print(c)
    for i in df.values: #convert DF to list of lists, i = 1 time step as an array
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    
    random.shuffle(sequential_data)

    buys = []
    sells = []
    for seq, target in sequential_data:
        if target==0:
            sells.append([seq, target])
        elif target==1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)
    
    #balance number of buys and sells
    lower=min(len(buys),len(sells))
    buys=buys[:lower]
    sells=sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data)

    #split into x (data for 1 time step)  and y (label)
    x=[]
    y=[]
    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)
    
    return np.array(x), y


#merge data frames from all files
main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
for ratio in ratios:
    dataset = f"Datasets/crypto_data/{ratio}.csv"

    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close" : f"{ratio}_close", "volume":f"{ratio}_volume"}, inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    #print(df.head())
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT) 
#shift future column up by F_P_P minutes

main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))
print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(10))
#print(main_df.head())

#build sequences, balance data, normalize data, scale data

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

#preprocess_df(main_df)
train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

#print(f"train data: {len(train_x)} validation: {len(validation_x)}")
#print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
#print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy', 
    optimizer=opt,
    metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

#checkpoint object
#filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
#checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
filepath = "epoch_{epoch:02d}-val_accuracy_{val_accuracy:.3f}"
checkpoint = ModelCheckpoint("cryptomodels/{}_{}.hd5".format(NAME,filepath), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint]
)