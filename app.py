from flask import *
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

import cv2
import os
from collections import deque
from tensorflow.keras.models import load_model  # Import the load_model function
import numpy as np
from moviepy.editor import VideoFileClip

# Specify the height and width each video frame will be resized in our dataset
IMAGE_HEIGHT, IMAGE_WIDTH = 64,64

# specify the list containing the names of the classes used for training.
CLASSES_LIST = ['Jump', 'Pick', 'Push', 'Run', 'Stand', 'Turn', 'Walk']

# specify the number of frames of a video that will be fed to the model as one sequence
SEQUENCE_LENGTH = 20

model = load_model('LRCN_model___Date_Time_2023_10_28__04_42_34___Loss_0.625197172164917___Accuracy_0.8232971429824829.h5')

def adjust_brightness_and_gamma(image, brightness=20.0, gamma=0.4):
    # Adjust brightness
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    # Apply gamma correction
    gamma_correction = 1.0 / gamma
    image = cv2.pow(image / 255.0, gamma_correction)
    image = (image * 255).astype(np.uint8)

    return image


def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform action recognition on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not ok:
            break
        
        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)

    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()



@app.route('/')
def index():
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    return response

ALLOWED_EXTENSIONS = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

vid = ['sample']



@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        flash('No video file found', 'error')
        return redirect(url_for('index'))
    video = request.files['video']
    if video.filename == "":
        flash('No video file selected', 'error')
        return redirect(url_for('index'))
    if video and allowed_file(video.filename):
        video.filename = "Input_Videos.mp4"
        video.save('static/Input_Videos/' + video.filename)
        test_video_directory = 'static/Output_Videos'
        test_video_directory1 = 'static'
        os.makedirs(test_video_directory, exist_ok=True)
        vid[0] = video.filename[:-4]
        video_title = vid[0]
        input_video_file_path = f'{test_video_directory}/{video_title}.mp4'
        # store the video name in the session
        session['video_name'] = vid[0]
        video_name = vid[0]
        # construct the output video path
        output_video_file_path = f'{test_video_directory1}/{video_title}=Output.SeqLen{SEQUENCE_LENGTH}.mp4'
        # Implement PRG pattern
        return redirect(url_for('index'))
    return 'invalid file type'


@app.route('/predict',methods=['GET'])
def predict():
    test_video_directory = 'static/Input_Videos'
    test_video_directory1 = 'static/Processed_Videos'
    if 'video_name' in session:
        video_title = vid[0]
        input_video_file_path = f'{test_video_directory}/{video_title}.mp4'

        # construct the output video path
        output_video_file_path = f'{test_video_directory1}/{video_title}=Processed.SeqLen{SEQUENCE_LENGTH}.mp4'
        if  input_video_file_path != '' and output_video_file_path != '':
            # Perform action recognition on the Test video
            predict_on_video(input_video_file_path,output_video_file_path,SEQUENCE_LENGTH)

            # Save the processed video
            processed_video_clip = VideoFileClip(output_video_file_path, audio=False, target_resolution=(300, None))
            processed_video_path = "static/Output_Videos/processed_output.mp4"  # Choose a filename and extension
            processed_video_clip.write_videofile(processed_video_path)

            # Open the saved video with the default video player
            # import subprocess
            # subprocess.run(["start", "", processed_video_path], shell=True)
            
            return render_template('preview.html',video_name= vid[0]+".mp4")
    flash('No video uploaded for prediction', 'error')
    return redirect(url_for('index'))



if __name__ == "__main__":
    app.run(debug=True)
