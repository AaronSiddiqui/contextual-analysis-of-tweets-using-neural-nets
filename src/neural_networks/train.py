from keras.callbacks import EarlyStopping, ModelCheckpoint

"""Returns a model that has been compiled and trained. It also includes 
   early stopping and checkpointing to save the best model"""


def train_nn(model, model_path, x_train, y_train, x_val, y_val,
             loss="binary_crossentropy", epochs=5, batch_size=64):
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    # Stops the model if it doesn't improve after a certain number of epochs
    # (patience)
    earlyStopping = EarlyStopping(monitor="val_accuracy", patience=5,
                                  verbose=1, mode="max")
    # Saves the model with the best validation accuracy
    checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy",
                                 save_best_only=True, mode="max", verbose=1)

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
              batch_size=batch_size, callbacks=[earlyStopping, checkpoint],
              verbose=2)

    return model
