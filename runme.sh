#!/usr/bin/env bash
pip install -r requirements.txt
mkdir "models"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 -o models/
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001 -o models/
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o models/
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o models/
#python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o models/ --precisions FP16
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o models/
#python src/driver.py -f models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml -l models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -it video -d CPU -debug headpose gaze