import numpy as np
import cv2
import ffmpeg
import sys
import os

def stream_video(input_url, out_folder, width, height, img_counter=1):
    """
    Stream video from a given input URL using ffmpeg and display it with OpenCV.

    This function opens a video stream from the specified input URL, decodes it using
    ffmpeg to raw video frames, and displays these frames using OpenCV. The function
    continues streaming until the video feed ends or is manually terminated.

    Parameters:
    - input_url : str
        The URL of the video stream to open. This can be any valid ffmpeg input, 
        such as a file path, RTSP, or UDP stream URL.
    - width : int
        The width of the video frames to be displayed.
    - height : int
        The height of the video frames to be displayed.

    Note:
    - To exit the video stream display, press 'q' while the OpenCV window is focused.
    """
    #cv2.namedWindow("Video Stream")

    process1 = (
        ffmpeg
        .input(input_url)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    image_counter = img_counter
    print(image_counter)
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

        file_name = out_folder + "/" + "image" + str(image_counter) + ".jpeg"

        cv2.imwrite(file_name, in_frame)
        image_counter += 1
        #if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            #break
        
    print(image_counter)
    #process1.wait()
    #cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Get command line arguments
    if (len(sys.argv)>1):
        in_file  = sys.argv[1]
        out_folder  = sys.argv[2]
    else: 
        in_file = os.environ['video-stream-url']
        out_folder  = os.environ['video-stream-image-folder']
    
    print("Starting process")
    #in_file = 'udp://127.0.0.1:23000'  # Example UDP input URL
    width = 3840  # Example width
    height = 2160  # Example height
    stream_video(in_file, out_folder, width, height)