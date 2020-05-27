#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

        ### TODO: Initialize any class variables desired ###
    def __init__(self , model, folder='', device='CPU', cpu_extension=None): 
        model_bin = folder + "/" + model + ".bin"
        model_xml = folder + "/" + model + ".xml"
        # Load Plugin
        self.plugin = IECore()
        if cpu_extension and "CPU" in device:                # Add a CPU extension, if applicable
            self.plugin.add_extension(cpu_extension, device)
        # Load Model
        self.load_model(model_bin, model_xml, device)
        print('Model: {}'.format(model))
        
    def load_model(self, model_bin, model_xml, device):
        '''
        1) Load model only once
        '''
        # Load Model
        net = IENetwork(model=model_xml, weights=model_bin)
        self.exec_net = self.plugin.load_network(net, device)
        # Input Output Blob
        self.input_blob = next(iter(net.inputs))
        self.output_blob = next(iter(net.outputs))
        # Input Shape [BxCxHxW] B=N
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        # print Model Input Output 
        print('Input: {}'.format(self.input_blob))
        print('Output: {}'.format(self.output_blob))
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        img = cv2.dnn.blobFromImage(image, size=(self.w, self.h))
        return img
        return

    def exec_net(self):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        image = self.preprocessing(image)
                self.exec_net.start_async(request_id=0, 
        inputs={self.input_blob: image})
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_net.requests[0].wait(-1)
        return status
    

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_net.requests[0].outputs[self.output_blob]
    
