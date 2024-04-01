from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import os
from photomosaic import Photomosaic
import tempfile
import cv2
import io

app = Flask(__name__)
#socketio = SocketIO(app)
# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

upload_percent = '0/100'

@app.route('/', methods=['GET', 'POST'])
def upload_form():
    temp_dir = tempfile.TemporaryDirectory()
    target_file_path = ''
    tile_file_path = []

    
    if request.method == 'POST':
        print(request.form.get('hor_tile)'))
        # Handle multiple image upload
        hor_tiles = int(request.form['hor_tile'])
        ver_tiles = int(request.form['ver_tile'])
        tile_opacity = int(request.form['slider'])

        print(tile_opacity)
        print(ver_tiles)
        print('--')
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

        photomosaic = Photomosaic(target_file_path, tile_file_path, hor_tiles, ver_tiles, tile_opacity)

        for tiles_done in photomosaic.transform():
            print(tiles_done)
        
        transformed_img_path = photomosaic.save_image(temp_dir.name)
        print(transformed_img_path)
        return send_file(transformed_img_path, as_attachment=True)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5100)
    #socketio.run(app)