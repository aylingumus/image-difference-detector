# import the necessary packages
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from skimage.metrics import structural_similarity
import imutils
import cv2
import base64
import os

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
Bootstrap(app)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_images(original_image, modified_image):
	# load the two input images
	imageA = cv2.imread(original_image)
	imageB = cv2.imread(modified_image)
	# convert the images to grayscale
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = structural_similarity(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	print("SSIM: {}".format(score))

	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)

	# loop over the contours
	for c in contours:
		# compute the bounding box of the contour and then draw the
		# bounding box on both input images to represent where the two
		# images differ
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
	with app.app_context():
		retval_imageA, buffer_imageA = cv2.imencode('.jpg', imageA)
		retval_imageB, buffer_imageB = cv2.imencode('.jpg', imageB)

	return buffer_imageA, buffer_imageB

@app.route("/")
@app.route('/hello')
def hello():
	return render_template('hello.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
	if request.method == 'POST':
		if len(request.files) == 0:
			return redirect(request.url)
		image_first = request.files['image_first']
		image_second = request.files['image_second']
		images = [image_first, image_second]
		filenames = []
		for image in images:
			if image.filename == '':
				flash("Please select all image files!", "error")
				return redirect(request.url)
			if image and allowed_file(image.filename):
				filename = secure_filename(image.filename)
				image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			filenames.append(filename)
		return redirect(url_for('show_diff', filenames=filenames))
	else:
		return render_template('upload_image.html')

@app.route('/show_diff', methods=['GET', 'POST'])
def show_diff():
	if request.method == "GET":
		image_first = os.path.join(app.config['UPLOAD_FOLDER'], request.args.getlist('filenames')[0])
		image_second = os.path.join(app.config['UPLOAD_FOLDER'], request.args.getlist('filenames')[1])
		buffer_imageA, buffer_imageB = process_images(image_first, image_second)
		with app.app_context():
			imageA = base64.b64encode(buffer_imageA.tobytes()).decode("utf-8")
			imageB = base64.b64encode(buffer_imageB.tobytes()).decode("utf-8")
			images = [imageA, imageB]
		return render_template('image_difference.html', images=images)
	else:
		return render_template('image_difference.html')

if __name__ == "__main__":
	app.run()
