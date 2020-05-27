"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from imutils.video import FPS
from utils.inference import OpenVinoNetwork
from utils.visualize import writetext

#deep sort
from tools.mytracking import MyTracking

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="/home/workspace/resources/Pedestrian_Detect_2_1_1.mp4.")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    client = None

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###

    ### TODO: Handle the input stream ###

    ### TODO: Loop until stream is over ###

        ### TODO: Read from the video capture ###

        ### TODO: Pre-process the image as needed ###

        ### TODO: Start asynchronous inference for specified request ###

        ### TODO: Wait for the result ###

            ### TODO: Get the results of the inference request ###

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            
            
   
       
###----------------------------------------------------------------------------------###
###    Function                                                                      ###
###----------------------------------------------------------------------------------###
def scaling(x, xmin, xmax, min_scale = 0.4, max_scale = 1):
    """
    customize result (minimum and maximum) scaling 
    """
    y = (x-xmin)/(xmax-xmin)*(max_scale-min_scale) + min_scale
    if x>xmax:
        y = max_scale
    return y 
#----------------------------------------------------------------------------------
def get_center(boxes):
    centers = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box.astype(int)
        center = (xmin, ymax)
        centers.append(center)
    return centers
#----------------------------------------------------------------------------------
def cal(M, points):
    arrPointsRes = []
    for point in points:
        px, py = point
        arrPoint = [[px], [py], [1]]
        arrRes = np.dot(M, arrPoint)
        rx = int(arrRes[0]/arrRes[2])
        ry = arrRes[1]/arrRes[2]
        arrPointsRes.append((int(rx), int(ry)))
    return arrPointsRes
#----------------------------------------------------------------------------------
class MovingAvgArr:
    def __init__(self, lenArr):
        self.arr = []
        self.length = lenArr
        
    def add_get(self, value):
        self.arr.append(value)
        if len(self.arr) > self.length:
            self.arr = self.arr[-(self.length-1):]
            current_count = int(round(np.mean(self.arr)))
        else:
            current_count = self.arr[-1]
        return current_count
        
#----------------------------------------------------------------------------------
class uniqueId:
    
    def __init__(self):
        self.arrId = {}
        self.counter = 1
        
    def add_get_id(self,currentId):
        
        try:
            realId = self.arrId[currentId]
        except:
            self.arrId[currentId] = self.counter
            realId = self.counter
            self.counter+= 1
        return realId

def main():
    
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
    # Parameter 

    source = 'input/Pedestrian_Detect_2_1_1.mp4'
    #source = 0
    nSkipFrame = 3
    # Visualization
    color = (0,255,0)
    width = 640
    height = 480
    # MATRIX TRANSFORMATION 
    
    M = np.array([[ 3.94063590e+00,  1.47607306e+01, -2.95214611e+03],
       [-7.94977438e-01,  1.03347067e+01,  3.97488719e+01],
       [ 4.70341705e-04,  7.43618778e-03,  1.00000000e+00]])
    #----------------------------------------------------------------------
    # Init Counter 
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    frameNow = nSkipFrame
    ma_people = MovingAvgArr(700)
    ma_sosdis = MovingAvgArr(3000)

    arrCount = []
    # Init Model
    person_det = OpenVinoNetwork('Pedestrian_Detect_2_1_1.mp4', folder='resources', device="CPU")
    # Deepsort
    if isTracking:
        mt = MyTracking()
        mt.init_deepSort()
        uId = uniqueId()
    
    # Init Video
    cap = cv2.VideoCapture(source)
    fps = FPS().start()
    #----------------------------------------------------------------------
    # saveVideo
    if saveVid:
        vid= cv2.VideoWriter_fourcc('M','J','P','G')
        vout= cv2.VideoWriter("Pedestrian_Detect_2_1_1.mp4", vid, 20.0, (890, 480))
        
    #----------------------------------------------------------------------------------        
    while cap.isOpened():
        try:
            ts = time.time()
            ret, frame = cap.read()
        except:
            pass
        if ret:
            #----------------------------------------------------------------------
            # Main Detection 
            #----------------------------------------------------------------------
            person_det.async_inference(frame)
            if person_det.wait() == 0:
                if frameNow >= nSkipFrame:
                    frameNow = 0
                    output = person_det.get_output()
                    output = output[0][:,np.where(output[0][0][:,1]==1)]
                    boxes, scores, boxes_W_H = post_detection(output, frame.shape, confident=.3)
                    # end of detection
                    if isTracking:
                        tracker,frame = mt.update_tracking(frame, boxes_W_H, scores)
            frameNow +=1
            #----------------------------------------------------------------------
            # If tracking
            if isTracking:
                boxes = []
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    personId = uId.add_get_id(track.track_id)
                    box = track.to_tlbr().astype(int)
                    # end of tracking
                    boxes.append(box) 
                    cap.release()
    if saveVid:
        vout.release()
    cv2.destroyAllWindows()
                    
    ### Topic "person": keys of "count" and "total" ###
    client.publish("person", json.dumps({"count": class_names}))
    client.publish("person", json.dumps({"duration": duration}))        
            
    ### Topic "person/duration": key of "duration" ###
            
            
    ### TODO: Send the frame to the FFMPEG server ###
    sys.stdout.buffer.write(out_frame)
    sys.stdout.flush

    client.disconnect()