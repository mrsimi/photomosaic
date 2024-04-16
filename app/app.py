from flask import Flask, render_template, request, send_file
import os
from photomosaic import Photomosaic
import tempfile
from utils import increase_count

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_form():
    temp_dir = tempfile.TemporaryDirectory()
    target_file_path = ''
    tile_file_path = []

    
    if request.method == 'POST':
        print(request.form.get('hor_tile)'))
        # Handle multiple image upload
        hor_tiles = int(request.form['hor_tile']) if request.form['hor_tile'] else 10
        ver_tiles = int(request.form['ver_tile']) if request.form['ver_tile'] else 10
        tile_opacity = int(request.form['slider']) if request.form['slider'] else 50

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

        transformed_img_path = Photomosaic(target_file_path, tile_file_path, hor_tiles, ver_tiles, tile_opacity).transform(temp_dir.name)
        
        #print(transformed_img_path)
        increase_count()
        return send_file(transformed_img_path, as_attachment=True)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))