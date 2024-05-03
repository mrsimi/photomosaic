from flask import Flask, render_template, request, send_file,send_from_directory
import os
import tempfile
from photomosaic import Photomosaic
import traceback
from photomosaic_fun import process_optimized


app = Flask(__name__)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')

# Custom error handler
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the exception
    app.logger.error(traceback.format_exc())

    # Redirect to the error page with status code 500 (Internal Server Error)
    return render_template('error.html', error_message=str(e)), 500

# Route for 404 (Page Not Found) error
@app.errorhandler(404)
def page_not_found_error(e):
    return render_template('404.html'), 404

# Route for 500 (Internal Server Error) error
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route('/', methods=['GET', 'POST'])
def upload_form():
    temp_dir = tempfile.TemporaryDirectory()
    target_file_path = ''
    tile_file_path = []
    

    if request.method == 'POST':
        divisions = int(request.form['slider']) if request.form['slider'] else 20
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
        
        print(f'Target file path {target_file_path}')
        print(f'Tile images: {tile_file_path}')

        #result_file = Photomosaic(tile_file_path, target_file_path, temp_dir.name, mosaic_size=40, divisions=40,tile_choice=5).process()
        result_file = process_optimized(target_file_path, tile_file_path,divisions,temp_dir.name)
        #result_file = result.save(f'{temp_dir.name}/download.jpg')
        print('result file ', result_file)
        return send_file(result_file, as_attachment=True)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))