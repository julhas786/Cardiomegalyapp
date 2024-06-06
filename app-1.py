import tempfile

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model
# Import necessary libraries
import os
from PIL import Image
from matplotlib import cm
import sys
import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
# os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Define function to generate Grad-CAM visualization for given image and CNN model
def grad_cam(model, image, layer_name):

    # Define model that generates both final model predictions as well as output of chosen layer
    # for i in model.layers:
    #     print(i.name)
    # print(model.inputs)
    # print(model.outputs)
    # print(model.get_layer(layer_name).output)
    grad_model = tf.keras.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    # Incoming image is singular example so expand dimensions to represent batch of size 1
    image_tensor = np.expand_dims(image, axis=0)

    # Cast image tensor to float32 type
    inputs = tf.cast(image_tensor, tf.float32)

    # Set up gradient tape to monitor intermediate variables and predictions
    with tf.GradientTape() as tape:

        # Extract activations from chosen layer and model's final predictions
        last_conv_layer_output, preds = grad_model(inputs)

        # Identify predicted class from final predictions
        pred_class = tf.argmax(preds[0])

        # Get output of predicted class from final layer
        class_channel = preds[:, pred_class]

    # Compute gradient of output with respect to chosen layer's output
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Reduce 2D gradients to 1D by averaging across height and width dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply 2D output map of chosen layer by 1D pooled gradients
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to be between 0 and 1 for better visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Return Grad-CAM heatmap
    return heatmap.numpy()

app = Flask(__name__)

# Custom deserialization for BinaryCrossentropy
def custom_binary_crossentropy(**kwargs):
    if 'fn' in kwargs:
        del kwargs['fn']
    return BinaryCrossentropy(**kwargs)

# Load the Keras model with custom objects
custom_objects = {'BinaryCrossentropy': custom_binary_crossentropy}
model = load_model('model2.h5', custom_objects=custom_objects)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # try:
    # Check if the request contains the file
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    # Get the file from the request
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save the image to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        file.save(temp_file.name)
        image_path = temp_file.name

    # Read the image file
    new_img = cv2.imread(image_path)

    # Preprocess the image
    new_img = new_img / 255.0
    new_img1 = cv2.resize(new_img, (224, 224))
    new_img = np.expand_dims(new_img1, axis=0)

    # Make prediction
    prediction = model.predict(new_img)
    # grad cam addition from my side
    layer_name = 'conv2d_13'
    grad_cam_image = grad_cam(model, new_img1, layer_name)
    # Enhance heatmap image for better visualization
    grad_cam_image = np.maximum(grad_cam_image, 0)
    grad_cam_image = np.minimum(grad_cam_image, 1)
    heatmap_colored = np.uint8(255 * grad_cam_image)

    # # Enhance heatmap image for better visualization
    # grad_cam_image = np.maximum(grad_cam_image, 0)
    # grad_cam_image = np.minimum(grad_cam_image, 1)
    # heatmap_colored = cm.jet(grad_cam_image)[:, :, :3]
    # heatmap_colored = np.uint8(255 * heatmap_colored)

    # Resize heatmap to original image size
    heatmap_resized = np.array(Image.fromarray(heatmap_colored).resize((new_img1.shape[1], new_img1.shape[0])))

    # Convert image and heatmap to 0-255 scale
    if new_img1.max() <= 1:
        new_img1 = (new_img1 * 255).astype('uint8')
    if heatmap_resized.max() <= 1:
        heatmap_resized = (heatmap_resized * 255).astype('uint8')

    # Superimpose heatmap on original image, with more weight on original image
    # superimposed_image = heatmap_resized * 0.4 + image * 0.6
    # superimposed_image = np.clip(superimposed_image, 0, 255).astype('uint8')
    # print(heatmap_resized)
    # Convert heatmap to a colored heatmap using a colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_MAGMA)

    # Overlay heatmap on the original image
    overlay = cv2.addWeighted(new_img1, 0.5, heatmap_colored, 0.5, 0)

    # Save the resulting image
    cv2.imwrite('./static/original.png', 255*new_img1)
    # cv2.imwrite("./static/gradcamheatmap.png",cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB))
    cv2.imwrite("./static/gradcamsuperimposed.png",overlay)
    show = "True"

    # Interpret the prediction
    result = "Positive: cardiomegaly" if np.argmax(prediction)==0 else "negative"
    return render_template('index.html', prediction=result, show=show)
    
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
