from keras.backend.tensorflow_backend import expand_dims, squeeze, ctc_batch_cost
from keras.models import Model
from keras.layers import Input, Lambda, ZeroPadding2D, Conv2D, LSTM, CuDNNLSTM, \
    Dense, ReLU, TimeDistributed, Dropout, BatchNormalization, Bidirectional
import tensorflow as tf


def ctc_lambda_func(args):
    labels, y_pred, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


def deepspeech(is_gpu=False, feature_cnt=128, output_dim=29, context=7, units=256, dropout_rates=[0.3, 0.3, 0.3]):
    # Input shape: (time, number of spectrogram features)
    input_tensor = Input([None, feature_cnt], name='X_input')
    # Reshape input to match Conv2D's input shape
    x = Lambda(expand_dims, arguments=dict(axis=-1))(input_tensor)
    x = ZeroPadding2D(padding=(context, 0))(x)
    receptive_field = (context * 2 + 1, feature_cnt)
    x = Conv2D(filters=units, kernel_size=receptive_field)(x)
    # Remove last dimension after convolution
    x = Lambda(squeeze, arguments=dict(axis=2))(x)
    # Apply activation and regulation
    x = ReLU(max_value=20)(x)
    x = Dropout(rate=dropout_rates[0])(x)

    # 2nd and 3rd groups of layers extract features from the context
    x = TimeDistributed(Dense(units))(x)
    x = ReLU(max_value=20)(x)
    x = Dropout(rate=dropout_rates[1])(x)

    x = TimeDistributed(Dense(units))(x)
    x = ReLU(max_value=20)(x)
    x = Dropout(rate=dropout_rates[2])(x)

    # LSTM handles long dependencies
    x = Bidirectional(CuDNNLSTM(units, return_sequences=True) if is_gpu
                      else LSTM(units, return_sequences=True), merge_mode='sum')(x)

    # Output at each time step probablity of each character
    output_tensor = TimeDistributed(
        Dense(output_dim, activation='softmax'), name='prediction_output')(x)

    model = Model(input_tensor, output_tensor)
    return model


def get_trainable_model(model):
    y_pred = model.outputs[0]
    model_input = model.inputs[0]

    labels = Input(shape=[None, ], dtype='int32')
    input_length = Input(shape=[1], dtype='int32')
    label_length = Input(shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, name='ctc')(
        [labels, y_pred, input_length, label_length])
    trainable_model = Model(
        inputs=[model_input, labels, input_length, label_length], outputs=loss_out)
    return trainable_model


def get_predictable_model(model):
    predictable_model = Model(inputs=[model.get_layer('X_input').input], outputs=[
                              model.get_layer('prediction_output').output])
    return predictable_model
