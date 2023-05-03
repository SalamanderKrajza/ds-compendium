import tensorflow as tf
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs_to_break_line=100):
        self.epochs_to_break_line = epochs_to_break_line

    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch}/{self.params["epochs"]}: training loss = {logs["loss"]:.4f}, validation loss = {logs["val_loss"]}', end='\r')
        if epoch%self.epochs_to_break_line==0:
            print("")
            
    def on_train_end(self, logs=None):
        print('\n', end='')
        super().on_train_end(logs)