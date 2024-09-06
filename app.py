from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, send_from_directory
import os
import shutil
import subprocess
import requests
from urllib.parse import urlparse
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import firebase_admin
from google.cloud import storage
from firebase_admin import credentials
from werkzeug.utils import secure_filename
from io import BytesIO
from pytube import YouTube # Import YouTube for downloading YouTube videos
from flask import Flask, request, jsonify, send_file
from urllib.parse import unquote
from flask import send_from_directory
from flask import Flask, request, jsonify, send_from_directory
import os
import requests
from pytube import YouTube
from pytube.exceptions import PytubeError

# Initialize Flask app
app = Flask(__name__)
CWD = os.getcwd()

# Configure app
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mpeg', 'webm', 'mov'}
app.config['YOLOV5_GITHUB_URL'] = 'https://github.com/ultralytics/yolov5.git'
app.config['YOLOV5_LOCAL_PATH'] = './yolov5'
app.config['TEMP_FOLDER'] = 'temp'
app.config['FIREBASE_BUCKET'] = 'cardio-1c22a.appspot.com'

# Ensure necessary folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['TEMP_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"cardio-1c22a-firebase-adminsdk-ampvt-e158e160d4.json"

# Initialize Firebase app
cred = credentials.Certificate(r"cardio-1c22a-firebase-adminsdk-ampvt-e158e160d4.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {'storageBucket': app.config['FIREBASE_BUCKET']})

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Function to check if YOLOv5 directory exists
def check_yolov5_exists():
    yolov5_folder = app.config['YOLOV5_LOCAL_PATH']
    return os.path.exists(yolov5_folder)

# Function to download YOLOv5 from GitHub
def download_yolov5_from_github():
    if check_yolov5_exists():
        print("YOLOv5 already installed. Skipping installation.")
        return

    try:
        subprocess.run(['git', 'clone', app.config['YOLOV5_GITHUB_URL'], app.config['YOLOV5_LOCAL_PATH']], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error cloning YOLOv5 repository: {str(e)}")

# Download YOLOv5 files before importing actual.py
download_yolov5_from_github()

from actual import main  # Import after Firebase initialization

# Function to check if a file is allowed based on its extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/get_videos', methods=['GET'])
def get_videos():
    videos = []
    for filename in os.listdir(CWD):
        if allowed_file(filename):
            videos.append({
                'url': f'/videos/{filename}',
                'file': filename
            })
    return jsonify(videos)

@app.route('/videos/<filename>')
def video(filename):
    return send_from_directory(CWD, filename)

@app.route('/process_query', methods=['POST'])
def process_query():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400
    video_file = request.files['video']
    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(CWD, filename)
        video_file.save(video_path)

        query = request.form.get('query')

        # Process the video
        video_output_path = main(video_path, query)
        if video_output_path:
            original_url, processed_url = upload_to_firebase(video_path, video_output_path)
            print("Uploaded to Firebase successfully")
            return jsonify({'success': True, 'result_video_url': processed_url})
        else:
            return jsonify({'success': False, 'error': 'Error processing video'}), 500

    return jsonify({'success': False, 'error': 'Invalid file format'}), 400

# Function to preprocess query and extract meaningful words
def preprocess_query(query):
    words = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    interrogative_terms = {'who', 'what', 'where', 'when', 'why', 'how', 'find'}
    meaningful_words = []
    for word, tag in pos_tag(words):
        if word not in stop_words and word not in interrogative_terms and (tag.startswith('N') or tag.startswith('V')):
            meaningful_words.append(word)
    return meaningful_words

# Function to remove local files and folders recursively
def remove_local_files(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# Function to sanitize filenames
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Function to upload videos to Firebase Storage and return their URLs
def upload_to_firebase(original_path, processed_path):
    bucket = storage_client.bucket(app.config['FIREBASE_BUCKET'])

    original_blob = bucket.blob('videos/' + os.path.basename(original_path))
    processed_blob = bucket.blob('videos/' + os.path.basename(processed_path))

    original_blob.upload_from_filename(original_path)
    processed_blob.upload_from_filename(processed_path)
    
    original_blob.make_public()
    processed_blob.make_public()

    original_url = original_blob.public_url
    processed_url = processed_blob.public_url
    
    print(f"Uploaded original video URL: {original_url}")
    print(f"Uploaded processed video URL: {processed_url}")

    return original_url, processed_url


# Modify index route to handle both local upload and URL submission
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        file = request.files.get('file')
        video_url = request.form.get('videoUrl')

        if (query and file and allowed_file(file.filename)) or (query and video_url):
            if file and allowed_file(file.filename):
                video_filename = sanitize_filename(file.filename)
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                file.save(video_path)
            elif video_url:
                try:
                    response = requests.post('/download_video', json={'url': video_url})
                    video_bytes = BytesIO(response.content)
                    video_path = os.path.join(app.config['TEMP_FOLDER'], 'downloaded_video.mp4')
                    with open(video_path, 'wb') as f:
                        f.write(video_bytes.read())
                except Exception as e:
                    print(f"Error downloading video: {str(e)}")
                    return jsonify({'error': 'Error downloading video'}), 500

            video_output_path = main(video_path, query)
            if video_output_path:
                original_url, processed_url = upload_to_firebase(video_path, video_output_path)
                print("Uploaded to Firebase successfully")
                return jsonify({'original_url': original_url, 'processed_url': processed_url})
            else:
                print("Error processing video")
                return jsonify({'error': 'Error processing video'}), 500
        else:
            print("Invalid query or file")
            return jsonify({'error': 'Please provide both query and video file or video URL'}), 400
    
    return render_template('index.html')

# Route for cleaning up local and Firebase storage
@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
        remove_local_files(app.config['UPLOAD_FOLDER'])
        remove_local_files(app.config['TEMP_FOLDER'])

        bucket = storage_client.bucket(app.config['FIREBASE_BUCKET'])
        blobs = bucket.list_blobs(prefix='videos/')
        for blob in blobs:
            blob.delete()

        flash('Cleanup successful', 'success')
    except Exception as e:
        flash(f"Error during cleanup: {str(e)}", 'error')

    return redirect(url_for('index'))

# Route for streaming the processed video
@app.route('/stream_video/<filename>')
def stream_video(filename):
    video_folder = app.config['UPLOAD_FOLDER']
    full_path = os.path.join(video_folder, filename)
    
    if not os.path.exists(full_path):
        return "Error: Video not found", 404

    return send_file(full_path, mimetype='video/mp4')

DOWNLOAD_DIRECTORY = 'downloads'

def ensure_download_directory():
    if not os.path.exists(DOWNLOAD_DIRECTORY):
        os.makedirs(DOWNLOAD_DIRECTORY)

# Ensure the download directory exists at the start of the application
ensure_download_directory()

@app.route('/download_video', methods=['POST'])
def download_video():
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Function to download video and handle filename uniqueness
        def download_file(url, filename):
            response = requests.get(url, stream=True)
            response.raise_for_status()

            filepath = os.path.join(DOWNLOAD_DIRECTORY, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Create directory if not exists

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

            return filename

        # Check if the URL is a YouTube URL
        if 'youtube.com' in url or 'youtu.be' in url:
            try:
                # Use yt-dlp to download the video from YouTube
                filename = 'result_video.mp4'
                counter = 1
                while os.path.exists(os.path.join(DOWNLOAD_DIRECTORY, filename)):
                    filename = f'result_video_{counter}.mp4'
                    counter += 1

                # yt-dlp command to download video
                command = ['yt-dlp', '--merge-output-format', 'mp4', '-o', os.path.join(DOWNLOAD_DIRECTORY, filename), url]
                result = subprocess.run(command, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"yt-dlp error: {result.stderr}")
                    return jsonify({'error': 'Error downloading YouTube video. Please try again later.'}), 500

            except Exception as e:
                print(f"Error downloading video: {str(e)}")
                return jsonify({'error': 'Error downloading video'}), 500

        else:
            # Handle non-YouTube URLs
            filename = 'result_video.mp4'
            counter = 1
            while os.path.exists(os.path.join(DOWNLOAD_DIRECTORY, filename)):
                filename = f'result_video_{counter}.mp4'
                counter += 1

            try:
                filename = download_file(url, filename)
            except requests.exceptions.RequestException as e:
                print(f"RequestException downloading video: {str(e)}")
                return jsonify({'error': 'Error downloading video. Check URL or try again later.'}), 500

        # Return the correct video path relative to DOWNLOAD_DIRECTORY
        return jsonify({'video_path': f'/serve_video/{filename}'})

    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return jsonify({'error': 'Error downloading video'}), 500

@app.route('/serve_video/<filename>')
def serve_video(filename):
    return send_from_directory(DOWNLOAD_DIRECTORY, filename)


# Main entry point for the application
if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = './uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)