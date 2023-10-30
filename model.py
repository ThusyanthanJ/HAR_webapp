import cv2
import os
from collections import deque
from tensorflow.keras.models import load_model  # Import the load_model function
import numpy as np
from moviepy.editor import VideoFileClip

# Specify the height and width each video frame will be resized in our dataset
IMAGE_HEIGHT, IMAGE_WIDTH = 64,64

# specify the list containing the names of the classes used for training.
CLASSES_LIST = ["Jump","Run","Walk"]

# specify the number of frames of a video that will be fed to the model as one sequence
SEQUENCE_LENGTH = 60

LRCN_model = load_model('LRCN_model__Date_Time_2023_09_21__10_03_02__Loss_0.6274334788322449__Accuracy_0.7642857432365417.h5')

def adjust_brightness_and_gamma(image, brightness=20.0, gamma=0.4):
    # Adjust brightness
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    # Apply gamma correction
    gamma_correction = 1.0 / gamma
    image = cv2.pow(image / 255.0, gamma_correction)
    image = (image * 255).astype(np.uint8)

    return image

def predict_on_video(video_file_path,output_file_path,SEQUENCE_LENGTH):
  """
  This function will perform action on a video using the LCRN model.
  Args:
    video_file_path: The path of the video stored in the disk on which the action recognition is to be performed
    output_file_path: The path where the output video with the predicted action being performed overlayed will be stored
    SEQUENCE_LENGTH: The fixed number of a video that can be passed to the model as one sequence
  """

  # Initialize the video capture object to read from the video
  video_reader = cv2.VideoCapture(video_file_path)

  # Get the width and height of the video
  original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
  original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Initialize the videowriter object to store the output video in the disk
  # Replace these placeholders with actual values
  fourcc_code = cv2.VideoWriter_fourcc(*'MP4V')  # Replace 'YOUR_FOURCC' with the correct fourcc code
  fps = video_reader.get(cv2.CAP_PROP_FPS)  # Replace with actual frame rate value
  frame_size = (original_video_width, original_video_height)  # Replace with actual frame dimensions

  video_writer = cv2.VideoWriter(output_file_path, fourcc_code, fps, frame_size)

  # Declare a queue to store video frame
  frames_queue = deque(maxlen = SEQUENCE_LENGTH)

  # Initialize a variable to store the predicted action being performed in the video
  predicted_class_name = ''

  # Iterate until the video is accessed successfully
  while video_reader.isOpened():

    # Read the frame
    ok, frame = video_reader.read()

    # check if frame is not read properly then break the loop
    if not ok:
      break
    frame = adjust_brightness_and_gamma(frame, brightness=1.5, gamma=1.5)
    # Resize the Frame to fixed Dimension
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT,IMAGE_WIDTH))

    # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
    normalized_frame = resized_frame/255

    # Appending the pre-processed frame into the frames list
    frames_queue.append(normalized_frame)

    # Check if the number of frames in the queue are equal to the fixed sequence length
    if len(frames_queue) == SEQUENCE_LENGTH:

      # Pass the normalized frames to the model and get the predicted probabilities
      predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue,axis=0))[0]

      # get the index of class with highest probalities
      predicted_label = np.argmax(predicted_labels_probabilities)

      # get the class name using the retrieve index
      predicted_class_name = CLASSES_LIST[predicted_label]

    # write predicted class name on top of the frame
    cv2.putText(frame,predicted_class_name,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # write the frame into the disk using the videowriter object
    video_writer.write(frame)

  # release the VC and VW objects
  video_reader.release()
  video_writer.release()

test_video_directory = 'static/videos'
test_video_directory1 = 'test_videos'
os.makedirs(test_video_directory,exist_ok=True)
video_title = 'test_video'
input_video_file_path = f'{test_video_directory}/{video_title}.mp4'

# construct the ouput video path
output_video_file_path = f'{test_video_directory1}/{video_title}=Output.SeqLen{SEQUENCE_LENGTH}.mp4'

# Perform action recognition on the Test video
predict_on_video(input_video_file_path,output_video_file_path,SEQUENCE_LENGTH)

# Save the processed video
processed_video_clip = VideoFileClip(output_video_file_path, audio=False, target_resolution=(300, None))
processed_video_path = "static/test_videos/processed_output.mp4"  # Choose a filename and extension
processed_video_clip.write_videofile(processed_video_path)

# Open the saved video with the default video player
# import subprocess
# subprocess.run(["start", "", processed_video_path], shell=True)
