from argparse import ArgumentParser

import logging
from input_feeder import InputFeeder
import constants


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
                        help="Provide the source of video frames. " + constants.VIDEO + " " + constants.WEBCAM + " | " + constants.IP_CAMERA + " | "+constants.IMAGE)
    parser.add_argument("-p", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="To see the visualization of different model outputs of each frame")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
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
    args = build_argparser().parse_args()
    logger = logging.getLogger()

    if (args.input_type == constants.VIDEO):
        extension = str(args.input).split('.')[1]
        if extension == 'mp4':

