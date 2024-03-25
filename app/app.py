from flask import Flask, render_template, request, send_file
from PIL import Image
from io import BytesIO
import os
from photomosaic import Photomosaicv2
import tempfile
import cv2
import io

app = Flask(__name__)
# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_form():
    temp_dir = tempfile.TemporaryDirectory()
    target_file_path = ''
    tile_file_path = []
    if request.method == 'POST':
        # Handle multiple image upload
        files = request.files.getlist('multi_files')
        for file in files:
            if file.filename == '':
                continue
            file_path = os.path.join(temp_dir.name, file.filename)
            file.save(file_path)
            tile_file_path.append(file_path)

        # Handle single image upload
        single_file = request.files['single_file']
        if single_file.filename != '':
            single_file_path = os.path.join(temp_dir.name, single_file.filename)
            single_file.save(single_file_path)
            target_file_path = single_file_path

        transformed_img_path = Photomosaicv2(target_file_path, tile_file_path, 10, 10).transform(temp_dir.name)
        print(transformed_img_path)
        return send_file(transformed_img_path, as_attachment=True)

    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5100)