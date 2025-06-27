import sys
import os
import uuid
import numpy as np
from types import SimpleNamespace
from PIL import Image
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from flask import Flask, request, send_file, make_response

# type: ignore
from flask_cors import CORS

from werkzeug.utils import secure_filename
from src.pyceph.pyceph import predict
from src.pyceph.CephImageBatch import CephImage
from src.pyceph.CLIConfig import load_inputs_defaults, set_torch_device
from src.pyceph.ModelWrapper import ModelWrapper

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def init_config(image_path):
    config_dict = load_inputs_defaults()
    config = SimpleNamespace(**config_dict)
    config.image_src = image_path
    config.image_folder = None
    config = set_torch_device(config)
    return config

@app.route("/process", methods=["POST"])
def process_image():
    try:
        if 'image' not in request.files:
            print("❌ No file in request")
            return {"error": "No file uploaded"}, 400

        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        print(f"✅ File received: {filename}")

        config = init_config(filepath)
        model = ModelWrapper(config, from_cli=False).load_model()

        ceph = CephImage(filepath)
        ceph.process(model, config)

        print("✅ Image processed")

        ceph.print_landmarks_and_mark_on_image()
        print("✅ Landmarks marked")

        # Convert image for return
        output_img = (ceph.image * 255).astype(np.uint8)
        pil_img = Image.fromarray(output_img)
        img_io = io.BytesIO()
        pil_img.save(img_io, 'JPEG', quality=100)
        img_io.seek(0)

        print("✅ Sending image back")
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        print("❌ Exception occurred:", e)
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

