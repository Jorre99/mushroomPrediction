# app.py

import os

from flask import Flask, request, make_response, jsonify, render_template, redirect, flash
from flask.helpers import url_for
from werkzeug.utils import secure_filename
from fastai.vision.all import *
from fastai.data.external import *


# codeblock below is needed for Windows path #############
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
##########################################################

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

learner1 = load_learner('mushroomNoMushroom.pkl')
learner2 = load_learner('toxic_safe.pkl')
learner3 = load_learner('mushroomNoMushroom.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:

            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename): 
            filename = secure_filename(file.filename)

            file.save('./static/' + filename)

            img = PILImage.create(file)
            
            pred1 = learner1.predict(img)
            print(pred1)
            # if you want a json reply, together with class probabilities:
            # return jsonify(str(pred1))
            # or if you just want the result
            # return {'success': pred1 }, 200

            # return redirect('/result/'+ pred1[0])
            # return redirect('/result/'+ filename)
            if pred1[0] == "mushroom":
                # invoke models
                perc1 = float(pred1[2][0])

                pred2 = learner2.predict(img)
                print(pred2)

                ###### TODO\\
                # add percentage for pred2

                return render_template("resultGood.html", 
                filename=filename, 
                variable1 = pred1[0], perc1 = perc1,
                variable2 = pred2[0])


            if pred1[0] == "noMushroom":
                return render_template("resultFalse.html", filename = filename)

    return render_template('home.html')