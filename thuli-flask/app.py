from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
from qrant_query import get_suggestions
from texttoimage import getImage
from flask import send_from_directory


app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "supersecretkey"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    dropdown_option = request.form.get('option')

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = process_file_and_option(file_path, dropdown_option)

        # Store result in session
        session['result_text'] = result

        return redirect(url_for('show_result'))
    else:
        flash('Allowed file types are: png, jpg, jpeg, gif')
        return redirect(request.url)

def process_file_and_option(image_path, selected_option):
    print(f'Processing file: {image_path} with option: {selected_option}')
    result = get_suggestions(image_path, selected_option)
    return result

@app.route('/result')
def show_result():
    result_text = session.get('result_text')  # Retrieve result from session
    image_path = session.get('generated_image_path')  # Retrieve image path if it was generated
    session.pop('generated_image_path', None)  # Clear the image path after it is shown
    if result_text:
        return render_template('result.html', result=result_text, image_path=image_path)
    else:
        flash('No result available')
        return redirect(url_for('upload_form'))

@app.route('/generate_image', methods=['POST'])
def generate_image():
    text = request.form['text']
    image_path = getImage(text)  # Get the image path from getImage
    session['generated_image_path'] = "processed-images/generated_outfit.png"  # Update the path for the image
    return redirect(url_for('show_result'))

@app.route('/processed-images/<filename>')
def serve_image(filename):
    return send_from_directory('processed-images', filename)


if __name__ == '__main__':
    app.run(debug=True)
