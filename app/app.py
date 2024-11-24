import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
from ultralytics import YOLO
import shutil

import zipfile


UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    upload_path = Path( app.config['UPLOAD_FOLDER'] )

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            base_filename = filename.rsplit('.', 1)[0] 
            file.save(os.path.join(upload_path, filename))
            file_path = upload_path / filename

            model = YOLO( Path('model/best.pt') )
            results = model(source=file_path)
            imgs_path = upload_path / base_filename
            results[0].save_crop(imgs_path)
            
            for crop in os.listdir(imgs_path / 'panel'):
                shutil.move(src=imgs_path/ 'panel' / crop, dst=imgs_path / crop )
            if not os.listdir(imgs_path / 'panel'):
                Path.rmdir(imgs_path / 'panel')

            if os.path.exists(file_path):
                os.remove(file_path)

            shutil.make_archive(base_name=base_filename, format='zip',root_dir=upload_path / base_filename )
            shutil.move(str ( base_filename) + '.zip', upload_path / Path(str ( base_filename) + '.zip' ))
            return redirect(url_for('download_file', name= str( base_filename ) + '.zip'))
        
    return render_template('index.html')

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

