import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# To hold the latest image data globally (not recommended for production)
uploaded_image = None

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    global uploaded_image  # Reference global image variable

    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    uploaded_image = Image.open(file.stream)  # Store the image globally

    # Generate summary using BLIP
    inputs = processor(uploaded_image, return_tensors="pt")
    out = model.generate(**inputs)

    # Decode the output without limiting length
    description = processor.decode(out[0], skip_special_tokens=True)

    return jsonify({
        'summary': description,
    })

if __name__ == '__main__':
    app.run(debug=True)
