{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6f84385a-8a47-45f5-abb6-94fec5b2b46e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading model...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model loaded.\n",
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:5001\n",
      "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "INFO:werkzeug:127.0.0.1 - - [28/Jan/2025 22:59:40] \"GET / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 732ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:127.0.0.1 - - [28/Jan/2025 23:00:06] \"POST /predict HTTP/1.1\" 200 -\n",
      "INFO:werkzeug:127.0.0.1 - - [28/Jan/2025 23:00:06] \"GET /static/uploads/test_image1.jpg HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 411ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:127.0.0.1 - - [28/Jan/2025 23:00:19] \"POST /predict HTTP/1.1\" 200 -\n",
      "INFO:werkzeug:127.0.0.1 - - [28/Jan/2025 23:00:19] \"GET /static/uploads/test_image6.jpg HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 412ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:127.0.0.1 - - [28/Jan/2025 23:00:40] \"POST /predict HTTP/1.1\" 200 -\n",
      "INFO:werkzeug:127.0.0.1 - - [28/Jan/2025 23:00:40] \"GET /static/uploads/test_image5.jpg HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template, request, redirect, url_for\n",
    "import os\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing import image\n",
    "import numpy as np\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the model\n",
    "print(\"Loading model...\")\n",
    "model = tf.keras.models.load_model('C:/Users/Anurag/Projects/WEBAPP/vgg_net16.h5')\n",
    "print(\"Model loaded.\")\n",
    "\n",
    "# Define the upload folder\n",
    "UPLOAD_FOLDER = 'static/uploads/'\n",
    "app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER\n",
    "\n",
    "# Ensure the upload folder exists\n",
    "if not os.path.exists(UPLOAD_FOLDER):\n",
    "    os.makedirs(UPLOAD_FOLDER)\n",
    "\n",
    "# Preprocess the image for the model\n",
    "def preprocess_image(img_path):\n",
    "    img = image.load_img(img_path, target_size=(224, 224))  # VGG16 expects 224x224 images\n",
    "    img_array = image.img_to_array(img)\n",
    "    img_array = np.expand_dims(img_array, axis=0)\n",
    "    img_array /= 255.0  # Normalize the image\n",
    "    return img_array\n",
    "\n",
    "# Home page\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "# Handle image upload and prediction\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    if 'file' not in request.files:\n",
    "        return redirect(request.url)\n",
    "    file = request.files['file']\n",
    "    if file.filename == '':\n",
    "        return redirect(request.url)\n",
    "\n",
    "    # Save the uploaded file\n",
    "    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)\n",
    "    file.save(file_path)\n",
    "\n",
    "    # Preprocess the image and make a prediction\n",
    "    img_array = preprocess_image(file_path)\n",
    "    predictions = model.predict(img_array)\n",
    "\n",
    "    # Interpret the predictions\n",
    "    class_labels = ['Unhealthy', 'Healthy']  # Replace with your actual class labels\n",
    "    confidence = np.max(predictions) * 100\n",
    "    predicted_class = class_labels[np.argmax(predictions)]\n",
    "\n",
    "    # Check if confidence is less than 50% and reverse the classification\n",
    "    if confidence < 50:\n",
    "        predicted_class = class_labels[1 - np.argmax(predictions)]  # Flip the class\n",
    "        confidence = 100 - confidence  # Reverse the confidence percentage\n",
    "\n",
    "    return render_template('index.html', prediction=predicted_class, confidence=confidence, image_path=file_path)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=False, port=5001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cab4ad6c-274c-4274-91fa-2834bf3e2b65",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (cnn_env)",
   "language": "python",
   "name": "cnn_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
