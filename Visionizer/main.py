from flask import Flask, render_template, send_file, jsonify
from flask_frozen import Freezer
from huggingface_hub import InferenceClient
from PIL import Image
import io
import os
freezer = Freezer(app)

app = Flask(__name__)
HF_TOKEN = os.getenv("HF_TOKEN")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generateimages/<prompt>')
def generate(prompt):
    try:
        print(f"Received prompt: {prompt}")

        # Initialize the client with the model and token
        client = InferenceClient("digiplay/AbsoluteReality_v1.8.1", token="HF_TOKEN")

        print("Client initialized successfully.")

        # Generate the image
        image = client.text_to_image(prompt)
        print("Image generation completed.")
        
        # Save the image to an in-memory buffer
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        print(f"Error generating image for prompt '{prompt}': {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='35.200.171.190', port=8000, debug=True)
	freezer.freeze()
