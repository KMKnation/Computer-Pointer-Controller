'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import urllib

import cv2
import numpy as np

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
        else:
            self.input_file= input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        elif self.input_type=='ip-cam':
            # self.cap=cv2.VideoCapture(0)
            print('Using ip cam instead of webcam')
        else:
            self.cap=cv2.imread(self.input_file)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        global frame, ret
        while True:
            if self.input_type=='ip-cam':
                img_arr = np.array(bytearray(urllib.request.urlopen(self.input_file).read()), dtype=np.uint8)
                frame = cv2.imdecode(img_arr, -1)
                ret = True
                # cv2.imshow('IPWebcam', img)
            else:
                for _ in range(10):
                    ret, frame=self.cap.read()
            yield ret, frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

