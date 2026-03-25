import os
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

MODEL_FILE = "model.h5"

if not os.path.exists(MODEL_FILE):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1)) / 255.0
    x_test = x_test.reshape((10000, 28, 28, 1)) / 255.0

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save(MODEL_FILE)
else:
    model = load_model(MODEL_FILE)

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = 255 - image
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Digit Recognition</title>
</head>
<body>
    <h2>Upload Handwritten Digit Image</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            img = preprocess_image(img)
            pred = model.predict(img)
            prediction = int(np.argmax(pred))
    return render_template_string(HTML, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)