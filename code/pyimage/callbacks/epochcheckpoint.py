from tensorflow.keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
    def __init__(self, output_path, every=5, start_at=0):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model,
        # the number of epochs that must pass before the model is serialized to
        # dist and the current epoch value
        self.outputPath = output_path
        self.every = every
        self.intEpoch = start_at

    def on_epoch_end(self, epoch, logs=None):
        # check to see if the model should be serialized to disk
        if (self.intEpoch + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath, "epoch_{}.hdf5".format(self.intEpoch + 1)])
            self.model.save(p, overwrite=True)

        # increment the internal epoch counter
        self.intEpoch += 1