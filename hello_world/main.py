from flask import Flask, render_template, request, jsonify, redirect, url_for, helpers
from jinja2 import evalcontextfilter, Markup, escape
import numpy as xp
import os
from werkzeug import secure_filename
from PIL import Image
from io import StringIO
from io import BytesIO
import cv2
import predictor

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    img_file = request.files['img_file']
    if img_file and allowed_file(img_file.filename):
        filename = secure_filename(img_file.filename)
        orig_image = Image.open(img_file.stream)
        image = predictor.predict(orig_image)
        buf = BytesIO()
        # image.save(buf, 'JPEG')
        image.save(buf, 'PNG')
        response = helpers.make_response(buf.getvalue())
        # response.headers["Content-Type"] = "image/jpeg"
        response.headers["Content-Type"] = "image/png"
        return response
    else:
        return ''' <p>invalid extension</p> '''

@app.template_filter('verqs')
def verqs(value):
    return Markup(value + '?2')# + os.environ.get('CURRENT_VERSION_ID')
   

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python37_app]
