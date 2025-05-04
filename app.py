import os
import io
import requests
from flask import Flask, request, send_file, jsonify
from PIL import Image
import torch
from realesrgan import RealESRGAN

app = Flask(__name__)

# Download model if not exists
model_dir = 'weights'
model_path = os.path.join(model_dir, 'RealESRGAN_x4plus.pth')
model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth'

if not os.path.exists(model_path):
    print("🔽 Downloading RealESRGAN model weights...")
    os.makedirs(model_dir, exist_ok=True)
    r = requests.get(model_url, stream=True)
    with open(model_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("✅ Model downloaded successfully!")

# Load model (only once at startup)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights(model_path)

@app.route('/')
def home():
    return '🟢 RealESRGAN Image Enhancer API is Running!'

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        input_image = Image.open(image_file).convert('RGB')
        enhanced_image = model.predict(input_image)

        buf = io.BytesIO()
        enhanced_image.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
