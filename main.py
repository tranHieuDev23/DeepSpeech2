if __name__ ==  '__main__':
    from load_data import DatasetGenerator, load_data_from_folder
    from model import deepspeech, get_trainable_model
    from keras.callbacks import ModelCheckpoint

    n_mfcc = 128
    print("Loading training data...")
    X_train, X_train_length, y_train, y_train_length, item_train_count = load_data_from_folder('train', 16000, 128, 20000)
    generator_train = DatasetGenerator(X_train, X_train_length, y_train, y_train_length, item_train_count)
    print("Loading testing data...")
    X_test, X_test_length, y_test, y_test_length, item_test_count = load_data_from_folder('dev', 16000, 128, 1000)
    generator_test = DatasetGenerator(X_test, X_test_length, y_test, y_test_length, item_test_count)

    model = get_trainable_model(deepspeech(
        is_gpu=False, feature_cnt=n_mfcc, units=256))
    model.compile('adam', {'ctc': lambda y_true, y_pred: y_pred})
    model.summary()

    checkpoint_cb = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
    model.fit_generator(generator_train, epochs=120, validation_data=generator_test, callbacks=[checkpoint_cb])
    model.save_weights('model_weights.h5')
