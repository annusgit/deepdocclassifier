
"""
    This file provides a front-end interface for inference of our deepdocclassifier model
"""

from __future__ import print_function
from __future__ import division
import sys
# sys.path.append('../../src/')
from werkzeug.utils import secure_filename
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
import os
import sys
sys.path.insert(0, '../')
import logging
import numpy as np
from PIL import Image
from dataset import toTensor
from model import DeepDocClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# create model and make it ready for inference
model = DeepDocClassifier()
model.load_state_dict(torch.load('../model-219.pt', map_location='cpu'))
model.eval()
# criterion = nn.CrossEntropyLoss()
logging.info('model loaded')


# web functions in here
##############################################################################################
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tif'])
def allowed_filename(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS


# find out if a new image has been uploaded
def check_new_comer(folder_path):
    images = os.listdir(folder_path)
    for image in images:
        if image not in classified_images:
            return True, image
    return False, None

app = Flask(__name__)
@app.route('/interface/', methods=['GET', 'POST'])
def interface():
    # UPLOAD_FOLDER = '/home/annus/PycharmProjects/deepdocclassifier/src/web/uploads'
    UPLOAD_FOLDER = 'uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    basedir = os.path.abspath(os.path.dirname(__file__))
    # begin by checking if we have a new file uploaded by now
    evaluate, new_image = check_new_comer(folder_path=UPLOAD_FOLDER)
    scores = range(10); pred = None; evaluated = False
    if evaluate:
        evaluated = True
        classified_images.append(new_image) # because this will be evaluated now
        scores, pred = single_inference(image=os.path.join(UPLOAD_FOLDER, new_image))
        # return render_template('bar_chart.html', title='Predictions', max=1, labels=labels, values=scores)
        # return redirect(url_for('bar'))
    if request.method == 'POST':
        submitted_file = request.files['file']
        if submitted_file and allowed_filename(submitted_file.filename):
            filename = secure_filename(submitted_file.filename)
            submitted_file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('interface', filename=filename))
    print(new_image, classified_images, scores, pred)
    title = 'Prediction = {}'.format(reverse_labels[pred]) if evaluated \
        else 'No scores were evaluated \n because either an image was already processed or we have just begun!!!'
    return render_template('test.html', title=title, max=max(scores),
                           labels=[reverse_labels[x] for x in range(10)],
                           values=scores)


labels = {
            'ADVE'        : 0,
            'Email'       : 1,
            'Form'        : 2,
            'Letter'      : 3,
            'Memo'        : 4,
            'News'        : 5,
            'Note'        : 6,
            'Report'      : 7,
            'Resume'      : 8,
            'Scientific'  : 9
        }

reverse_labels = {v:k for k,v in labels.iteritems()}

# values = [
#     967.67, 1190.89, 1079.75, 1349.19,
#     2328.91, 2504.28, 2873.83, 4764.87,
#     4349.29, 6458.30, 9907, 16297
# ]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

@app.route('/bar')
def bar():
    bar_labels = labels
    global scores
    return render_template('bar_chart.html', title='Predictions', max=1, labels=bar_labels, values=scores)


#############################################################################################

classified_images = [] # list of all images that have been evaluated
transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

@torch.no_grad()
def single_inference(image):
    example_array = Image.open(image).resize((227, 227))
    example_array = np.asarray(example_array).astype(np.uint8)
    example_array = np.dstack((example_array, example_array, example_array))
    # print(example_array.shape)
    example_array = transform(toTensor(example_array)).unsqueeze(0)
    out_x, pred = model(example_array)
    return (F.softmax(out_x, dim=1).numpy()*1000).astype(np.float32).tolist()[0], int(pred.numpy())


if __name__ == '__main__':
    app.run(debug=True, port=8008, host='0.0.0.0')































