
"""
    This file provides a front-end interface for inference of our deepdocclassifier model
"""

from flask import Flask, flash, redirect, render_template, request, session, abort
app = Flask(__name__)


@app.route("/")
def getMember():
    return '?'

@app.route("/hello")
def hello():
    return '<h1> Hello World! </h1>'

@app.route('/interface/<string:name>')
def home(name):
    return render_template('test.html', name=name)


if __name__ == '__main__':
    app.run(debug=True, port=8008)
