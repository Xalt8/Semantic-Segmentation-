from keras.applications import VGG16
from keras.models import Model
import joblib
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Uniform dimensions for all images & masks 
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Get the imagenet weights, change to input shape to & make the model non-trainable
VGG_model = VGG16(weights = 'imagenet', include_top = False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
for layer in VGG_model.layers:
    layer.trainable = False


# Take the first few layers of VGG16 -> keeping the dimensions of image & mask arrays intact 
small_vgg = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)


# Load the x_train_scaled object
x_train_scaled = joblib.load("x_train_scaled.joblib")

# Get the features by calling predict on x_train data
vgg_feature_extractor = small_vgg.predict(x_train_scaled)


if __name__ =="__main__":
    print("In vgg.py main")
    # joblib.dump(vgg_feature_extractor, "vgg_feature_extractor.joblib")    