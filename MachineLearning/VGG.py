from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
import os
# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# VGG-16 instance
model = VGG16(weights='imagenet', include_top=True)

image = load_img('E:/人工智能/dataset/picture/sea.jpg', target_size=(224, 224))
image_data = img_to_array(image)

# reshape it into the specific format
image_data = image_data.reshape((1,) + image_data.shape)  # 1 picture

# prepare the image data for VGG
image_data = preprocess_input(image_data)

# using the pre-trained model to predict
prediction = model.predict(image_data)

# decode the prediction results
results = decode_predictions(prediction, top=3)

print(results)