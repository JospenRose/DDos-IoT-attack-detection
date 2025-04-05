from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.models import Model, Sequential
from save_load import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, matthews_corrcoef, cohen_kappa_score, hamming_loss, jaccard_score
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam


def confu_matrix(y_test, y_predict):
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    mcc = float(matthews_corrcoef(y_test, y_predict))
    kappa = float(cohen_kappa_score(y_test, y_predict))
    h_loss = hamming_loss(y_test, y_predict)
    jaccard = float(jaccard_score(y_test, y_predict))
    return [accuracy, precision, recall, f1, r2, mcc, kappa, h_loss, jaccard]



# 1. CNN for spatial feature extraction (AlexNet-like)
def create_alexnet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, (3,), activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=(1, ))(x)
    x = Conv1D(128, (3,), activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(1, ))(x)
    x = Conv1D(256, (3,), activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(1,))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    model = Model(inputs=inputs, outputs=x, name="AlexNet_Model")
    return model


# 2. LSTM for Sequential Data Processing
def create_lstm(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(32))(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    model = Model(inputs=inputs, outputs=x, name="LSTM_Model")
    return model


# 3. PSPNet-like Feature Extraction (Simplified CNN)
def create_pspnet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, (3,), activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=(1, ))(x)
    x = Conv1D(128, (3, ), activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(1, ))(x)
    x = Conv1D(256, (3, ), activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(1, ))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    model = Model(inputs=inputs, outputs=x, name="PSPNet_Model")
    return model


def TriGuardNet(x_train, y_train, x_test, y_test, epochs=55, batch_size=32, learning_rate=0.001):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    input_shape = x_train[1].shape

    alexnet_model = create_alexnet(input_shape)
    lstm_model = create_lstm(input_shape)
    pspnet_model = create_pspnet(input_shape)

    # Concatenation of features
    combined_features = tf.keras.layers.concatenate([alexnet_model.output, lstm_model.output, pspnet_model.output])
    x = Dense(256, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)  # Binary classification (Attack/Normal)

    # Define and compile model
    hybrid_model = Model(inputs=[alexnet_model.input, lstm_model.input, pspnet_model.input], outputs=output)
    hybrid_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    hybrid_model.summary()

    history = hybrid_model.fit([x_train, x_train, x_train], y_train, epochs=epochs, batch_size=batch_size, validation_data=([x_test, x_test, x_test], y_test))

    pred = hybrid_model.predict([x_test, x_test, x_test])
    y_pred = np.array([1 if i>0.5 else 0 for i in pred])

    met = confu_matrix(y_test, y_pred)

    return y_pred, met, history


def lstm(x_train, y_train, x_test, y_test, epoch=200):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    input_shape = x_train[1].shape
    num_classes = len(set(y_train))

    # Define the LSTM model
    model = Sequential([
        LSTM(16, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),

        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=epoch, batch_size=16, verbose=1)

    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)

    met = confu_matrix(y_test, pred)

    return pred, met


def CNN(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    num_classes = len(set(y_train))

    model = Sequential()

    model.add(Conv1D(32, (1, ), activation='relu', padding='same', input_shape=x_train[1].shape))
    model.add(MaxPooling1D(pool_size=(1, )))
    model.add(Conv1D(64, (1, ), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=120, batch_size=16)

    y_predict = model.predict(x_test)

    y_predict = np.argmax(y_predict, axis=1)

    return y_predict, confu_matrix(y_test, y_predict)


def dnn(x_train, y_train, x_test, y_test):
    input_shape = x_train[1].shape

    model = Sequential([
        Flatten(input_shape=(input_shape)),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1)
    pred = model.predict(x_test)
    y_predict = np.array([1 if i>0.5 else 0 for i in pred])

    return y_predict, confu_matrix(y_test, y_predict)


def AlexNet(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    input_shape = x_train[1].shape
    model = create_alexnet(input_shape)
    x = Dense(256, activation='relu')(model.output)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)  # Binary classification (Attack/Normal)

    # Define and compile model
    Alexnet_model = Model(inputs=model.input, outputs=output)
    Alexnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    Alexnet_model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1)

    pred = Alexnet_model.predict(x_test)
    y_pred = np.array([1 if i>0.5 else 0 for i in pred])

    met = confu_matrix(y_test, y_pred)

    return y_pred, met

