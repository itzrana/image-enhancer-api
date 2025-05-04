from flask import Flask, request, send_file
from PIL import Image
import io
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = Flask(__name__)

# Load model (weights should be in the 'weights' folder)
model_path = 'weights/RealESRGAN_x4plus.pth'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400

    image_file = request.files['image']
    input_image = Image.open(image_file).convert("RGB")

    # Convert to numpy array
    import numpy as np
    img_np = np.array(input_image)

    # Run Real-ESRGAN enhancement
    output, _ = upsampler.enhance(img_np, outscale=4)

    # Convert back to image
    output_image = Image.fromarray(output)

    img_io = io.BytesIO()
    output_image.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/')
def index():
    return 'Remini-style Image Enhancer API is running!'
