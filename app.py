import os
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from skimage import color, exposure
import skimage
import base64
print(skimage.__version__)



app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

PROCESSED_FOLDER = 'processed'
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Process and save pre-processed images
def process_and_save_images():
    subject_images = {
        '1': 'subject_images/subject_image_1.png',  # Path to subject image for filter 1
        '2': 'subject_images/subject_image_2.jpg',  # Path to subject image for filter 2
        '3': 'subject_images/subject_image_3.jpg',  # Path to subject image for filter 3
        '4': 'subject_images/subject_image_4.jpg'   # Path to subject image for filter 4
    }

    for filter_type, subject_image_path in subject_images.items():
        processed_img_path = f'{PROCESSED_FOLDER}/processed_img_{filter_type}.png'
        if os.path.exists(processed_img_path):
            print(f"Processed image for filter {filter_type} already exists. Skipping...")
            continue

        if not os.path.exists(subject_image_path):
            continue

        subject_img = cv2.imread(subject_image_path)
        processed_img = process_image(subject_img, filter_type)
        
        cv2.imwrite(processed_img_path, processed_img)

        print(f"Processed image for filter {filter_type} saved.")

# Function to process images based on filter type
def process_image(img, filter_type):
    if filter_type == '1':
        return lowpass_gaussian_filter(img)
    elif filter_type == '2':
        return lowpass_butterworth_filter(img)
    elif filter_type == '3':
        return highpass_laplacian_filter(img)
    elif filter_type == '4':
        return histogram_matching(img)


# Route to serve both original and processed images
@app.route('/processed_images/<filter_type>', methods=['GET'])
def get_processed_image(filter_type):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    original_image_path = f'subject_images/subject_image_{filter_type}'
    original_image = None

    for extension in ALLOWED_EXTENSIONS:
        if os.path.exists(f'{original_image_path}.{extension}'):
            original_image_path = f'{original_image_path}.{extension}'
            original_image = cv2.imread(original_image_path)
            break

    if original_image is None:
        return jsonify({'error': 'Original image not found'}), 404

    processed_img = process_image(original_image, filter_type)

    _, original_encoded = cv2.imencode('.png', original_image)
    _, processed_encoded = cv2.imencode('.png', processed_img)

    original_base64 = base64.b64encode(original_encoded).decode('utf-8')
    processed_base64 = base64.b64encode(processed_encoded).decode('utf-8')

    return jsonify({
        'original_image': original_base64,
        'processed_image': processed_base64
    })

# Route to process uploaded image
@app.route('/process/<filter_type>', methods=['POST'])
def process_uploaded_image(filter_type):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img = Image.open(file.stream)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    processed_img = process_image(img, filter_type)

    _, img_encoded = cv2.imencode('.png', processed_img)
    img_bytes = BytesIO(img_encoded.tobytes())

    return send_file(img_bytes, mimetype='image/png')

# Function to apply lowpass Gaussian filter
def lowpass_gaussian_filter(img):
    return cv2.GaussianBlur(img, (15, 15), 0)

# Function to apply lowpass Butterworth filter
def lowpass_butterworth_filter(img):
	# Convert the image to grayscale (assuming it's RGB)
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Convert to float32 and normalize to 0-1 range
  img_float32 = gray_img.astype(np.float32) / 255.0

  rows, cols = img_float32.shape[:2]
  crow, ccol = rows // 2, cols // 2
  d0 = 30
  n = 2

  # Create Butterworth filter mask
  butterworth_filter = np.zeros((rows, cols), np.float32)
  for i in range(rows):
    for j in range(cols):
      d = np.sqrt((i - crow)**2 + (j - ccol)**2)
      butterworth_filter[i, j] = 1 / (1 + (d / d0)**(2*n))

  # Perform frequency domain filtering
  img_dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
  img_dft_shift = np.fft.fftshift(img_dft)
  img_dft_shift[:,:,0] = img_dft_shift[:,:,0] * butterworth_filter
  img_dft_shift[:,:,1] = img_dft_shift[:,:,1] * butterworth_filter
  img_idft_shift = np.fft.ifftshift(img_dft_shift)
  img_back = cv2.idft(img_idft_shift)

  # Get magnitude spectrum (optional)
  img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

  # Normalize and convert back to uint8 (optional for display)
  img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

  return img_back


# Function to apply highpass Laplacian filter
def highpass_laplacian_filter(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)


# Function to apply histogram matching
def histogram_matching(source_image):
    reference_image_path = 'subject_images/reference_image_1.jpg'
    reference_image = cv2.imread(reference_image_path)

    # Calculate histograms for each channel
    source_hist_b = cv2.calcHist([source_image], [0], None, [256], [0, 256])
    source_hist_g = cv2.calcHist([source_image], [1], None, [256], [0, 256])
    source_hist_r = cv2.calcHist([source_image], [2], None, [256], [0, 256])

    reference_hist_b = cv2.calcHist([reference_image], [0], None, [256], [0, 256])
    reference_hist_g = cv2.calcHist([reference_image], [1], None, [256], [0, 256])
    reference_hist_r = cv2.calcHist([reference_image], [2], None, [256], [0, 256])

    # Normalize histograms
    source_hist_norm_b = source_hist_b / np.sum(source_hist_b)
    source_hist_norm_g = source_hist_g / np.sum(source_hist_g)
    source_hist_norm_r = source_hist_r / np.sum(source_hist_r)

    reference_hist_norm_b = reference_hist_b / np.sum(reference_hist_b)
    reference_hist_norm_g = reference_hist_g / np.sum(reference_hist_g)
    reference_hist_norm_r = reference_hist_r / np.sum(reference_hist_r)

    # Cumulative distribution function (CDF) calculation
    source_cdf_b = np.cumsum(source_hist_norm_b)
    source_cdf_g = np.cumsum(source_hist_norm_g)
    source_cdf_r = np.cumsum(source_hist_norm_r)

    reference_cdf_b = np.cumsum(reference_hist_norm_b)
    reference_cdf_g = np.cumsum(reference_hist_norm_g)
    reference_cdf_r = np.cumsum(reference_hist_norm_r)

    # Create mapping functions for each channel
    mapping_b = np.zeros(256)
    mapping_g = np.zeros(256)
    mapping_r = np.zeros(256)

    for i in range(256):
        mapping_b[i] = np.argmax(reference_cdf_b >= source_cdf_b[i])
        mapping_g[i] = np.argmax(reference_cdf_g >= source_cdf_g[i])
        mapping_r[i] = np.argmax(reference_cdf_r >= source_cdf_r[i])

    # Apply mappings to source image
    matched_image = cv2.merge((
        mapping_b[source_image[:,:,0]],
        mapping_g[source_image[:,:,1]],
        mapping_r[source_image[:,:,2]]
    ))

    # Convert back to uint8
    matched_image = np.uint8(matched_image)

    return matched_image


if __name__ == '__main__':
    process_and_save_images()  # Process and save pre-processed images
    app.run(debug=True, port=5000)
