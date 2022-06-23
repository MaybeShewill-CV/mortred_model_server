#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 22-6-22 上午12:12
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/MMAiServer
# @File    : test_yolov5_detection_server.py
# @IDE: PyCharm
"""
Test yolov5 detection server
"""
import argparse
import os
import os.path as ops
import requests
import base64
import json
import hashlib
import time

import locust


def test_yolov5_detection_server_once():
    """

    :return:
    """
    src_image_path = '../demo_data/model_test_input/object_detection/bus.jpg'
    assert ops.exists(src_image_path), '{:s} not exist'.format(src_image_path)
    with open(src_image_path, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data)

    # url = 'http://maybeshewill-cv.natapp1.cc/morted_ai_server_v1/obj_detection/yolov5'
    url = 'http://localhost:8094/morted_ai_server_v1/obj_detection/yolov5'
    task_id = src_image_path + str(time.time())
    m2 = hashlib.md5()
    m2.update(task_id.encode())
    task_id = m2.hexdigest()
    post_data = {
        'img_data': base64_data.decode(),
        'req_id': task_id,
    }

    try:
        resp = requests.post(url=url, data=json.dumps(post_data))
        output = json.loads(resp.text)
        print(output)
    except Exception as e:
        print(e)


class ClientBehavior(locust.TaskSet):
    """
    simulate client
    """
    def on_start(self):
        """

        :return:
        """
        print('client start ...')

    def on_stop(self):
        """

        :return:
        """
        print('client stop ...')

    @locust.task(1)
    def test_yolov5_classification_server(self):
        """

        :return:
        """
        src_image_path = '../demo_data/model_test_input/object_detection/bus.jpg'
        with open(src_image_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data)

        # url = 'http://maybeshewill-cv.natapp1.cc/morted_ai_server_v1/obj_detection/yolov5'
        url = 'http://localhost:8094/morted_ai_server_v1/obj_detection/yolov5'
        task_id = src_image_path + str(time.time())
        m2 = hashlib.md5()
        m2.update(task_id.encode())
        task_id = m2.hexdigest()
        post_data = {
            'img_data': base64_data.decode(),
            'req_id': task_id,
        }

        resp = self.client.post(url, data=json.dumps(post_data))
        # resp = self.client.post(url)
        if resp.status_code == 200:
            print(resp.text)
        else:
            print('request failed')


class WebBehavior(locust.HttpUser):
    """
    simulate server
    """
    tasks = [ClientBehavior]
    min_wait = 10
    max_wait = 40


if __name__ == '__main__':
    """
    main func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single', help='run mode')
    parser.add_argument('-u', type=int, default=20)
    parser.add_argument('-n', type=int, default=1000)
    parser.add_argument('-r', type=int, default=5)
    parser.add_argument('-t', type=str, default='5m')
    args = parser.parse_args()

    if args.mode == 'single':
        for i in range(1000):
            test_yolov5_detection_server_once()
    elif args.mode == 'locust':
        command = 'locust -f ./server/test_yolov5_detection_server.py ' \
                  '--host=http://localhost:8094/morted_ai_server_v1/obj_detection/yolov5 --headless ' \
                  '-u {:d} -r {:d} -t {:s}'.format(args.u, args.r, args.t)
        os.system(command=command)
    else:
        raise ValueError('unknown mode')

