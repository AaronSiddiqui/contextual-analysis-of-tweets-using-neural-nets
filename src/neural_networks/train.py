from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

"""Returns a model that has been compiled and trained. It also includes 
   early stopping and checkpointing to save the best model"""


def train_nn(model, model_path, x_train, y_train, x_val, y_val,
             loss, epochs=5, batch_size=64):
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    # Stops the model if it doesn't improve after a certain number of epochs
    # (patience)
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=5,
                                   verbose=1, mode="max")
    # Saves the model with the best validation accuracy
    checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy",
                                 save_best_only=True, mode="max", verbose=1)

    # Returns the validation loss and accuracy
    save_evaluation = SaveEvaluation((x_val, y_val), model_path)

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
              batch_size=batch_size, callbacks=[early_stopping, checkpoint,
                                                save_evaluation], verbose=2)

    return model


"""Custom callback to save validation loss and accuracy during training"""


class SaveEvaluation(Callback):
    def __init__(self, test_data, path):
        self.test_data = test_data
        self.path = path[:-3] + "_evaluation.txt"  # evaluation file

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        to_write = f"val_loss: {loss} \tval_acc: {acc}\n"

        # Appends the information to the evaluation file
        with open(self.path, "a") as f:
            f.write(to_write)
