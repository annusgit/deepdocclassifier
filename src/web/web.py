
"""
    This file provides a front-end interface for inference of our deepdocclassifier model
"""

from __future__ import print_function
from __future__ import division
from werkzeug.utils import secure_filename
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
app = Flask(__name__)
import os
import logging

@app.route("/")
def getMember():
    return '?'

@app.route("/hello")
def hello():
    return '<h1> Hello World! </h1>'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tif'])
def allowed_filename(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.route('/interface/', methods=['GET', 'POST'])
def interface():
    UPLOAD_FOLDER = '/home/annus/PycharmProjects/deepdocclassifier/src/web/uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    basedir = os.path.abspath(os.path.dirname(__file__))
    # file = request.file('image')
    # f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    # file.save(f)
    if request.method == 'POST':
        submitted_file = request.files['file']
        if submitted_file and allowed_filename(submitted_file.filename):
            filename = secure_filename(submitted_file.filename)
            submitted_file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('interface', filename=filename))
    return render_template('test.html', name='annus')


@app.route('/upload', methods=['POST'])
def upload():
    UPLOAD_FOLDER = os.path.basename('uploads')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    file = request.file('image')
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8008)











