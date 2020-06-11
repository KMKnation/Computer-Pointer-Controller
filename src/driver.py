from argparse import ArgumentParser

import logging
from input_feeder import InputFeeder
import constants
import os
from face_detection import Face_Model
from facial_landmarks_detection import Landmark_Model
from gaze_estimation import Gaze_Estimation_Model
from head_pose_estimation import Head_Pose_Model
from mouse_controller import MouseController
import cv2
import imutils

def imshow(windowname, frame, width=None):

    if width == None:
        width = 400

    frame = imutils.resize(frame, width=width)
    cv2.imshow(windowname, frame)

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--face", required=True, type=str,
                        help="Path to .xml file of Face Detection model.")
    parser.add_argument("-l", "--landmarks", required=True, type=str,
                        help="Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Provide the source of video frames. " + constants.VIDEO + " " + constants.WEBCAM + " | " + constants.IP_CAMERA + " | " + constants.IMAGE)
    parser.add_argument("-view", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="To see the visualization of different model outputs of each frame")
    parser.add_argument("-ld", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="linker libraries if have any")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Provide the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")

    return parser


if __name__ == '__main__':
    arg = '-f ../models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml -l ../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp ../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it video -d CPU'.split(' ')
    args = build_argparser().parse_args(arg)
    logger = logging.getLogger()

    feeder = None
    if args.input_type == constants.VIDEO or args.input_type == constants.IMAGE:
        extension = str(args.input).split('.')[1]
        # if not extension.lower() in constants.ALLOWED_EXTENSIONS:
        #     logger.error('Please provide supported extension.' + str(constants.ALLOWED_EXTENSIONS))
        #     exit(1)

        if not os.path.isfile(args.input):
            logger.error("Unable to find specified video/image file")
            exit(1)

        feeder = InputFeeder(args.input_type, args.input)
    elif args.input_type == constants.IP_CAMERA:
        if not str(args.input).startswith('http://'):
            logger.error('Please provide ip of server with http://')
            exit(1)

        feeder = InputFeeder(args.input_type, args.input)
    elif args.input_type == constants.WEBCAM:
        feeder = InputFeeder(args.input_type)

    mc = MouseController("medium", "fast")

    feeder.load_data()

    face_model = Face_Model(args.face, args.device, args.cpu_extension)
    face_model.check_model()

    landmark_model = Landmark_Model(args.landmarks, args.device, args.cpu_extension)
    landmark_model.check_model()

    gaze_model = Gaze_Estimation_Model(args.gazeestimation, args.device, args.cpu_extension)
    gaze_model.check_model()

    head_model = Head_Pose_Model(args.headpose, args.device, args.cpu_extension)
    head_model.check_model()



    face_model.load_model()
    logger.info("Face Detection Model Loaded...")
    landmark_model.load_model()
    logger.info("Landmark Detection Model Loaded...")
    gaze_model.load_model()
    logger.info("Gaze Estimation Model Loaded...")
    head_model.load_model()
    logger.info("Head Pose Detection Model Loaded...")
    print('Loaded')


    frame_count = 0
    for ret, frame in feeder.next_batch():
        if not ret:
            break
        frame_count +=1

        # if frame_count % 5 == 0:
        #     imshow("video", frame)
        imshow("video", frame)

        crop_face, box = face_model.predict(frame.copy())

        if crop_face is None:
            logger.error("Unable to detect the face.")
            continue

        imshow("face", crop_face, width=100)

        (lefteye_x, lefteye_y), (righteye_x, righteye_y), eye_coords, left_eye, right_eye = landmark_model.predict(crop_face.copy(), eye_surrounding_area=10)
        # imshow("left_eye", left_eye, width=100)
        # imshow("right_eye", right_eye, width=100)


        key = cv2.waitKey(60)
        if key == 27:
            break

    cv2.destroyAllWindows()