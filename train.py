import dlib

# Load training data
training_data = "/Users/anuragraturi/PycharmProjects/detection/images"

# Set parameters for training
options = dlib.shape_predictor_training_options()
options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 4
options.be_verbose = True

# Train the model
dlib.train_shape_predictor(training_data, "/Users/anuragraturi/PycharmProjects/detection/models", options)
