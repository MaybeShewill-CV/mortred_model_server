#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 22-7-2 01:12
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/MMAiServer
# @File    : locust_performance.py
# @IDE: PyCharm
"""
locust pressure test    
"""
import base64
import hashlib
import json
import os
import time

import locust

# injected by test_server.py (locust mode); no file mutation at runtime
URL = os.environ.get('LOCUST_URL', '')
SRC_IMAGE_PATH = os.environ.get('LOCUST_IMG', '')


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
        def test_server(self):
            """

            :return:
            """
            with open(SRC_IMAGE_PATH, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data)

            task_id = SRC_IMAGE_PATH + str(time.time())
            m2 = hashlib.md5()
            m2.update(task_id.encode())
            task_id = m2.hexdigest()
            post_data = {
                'images': [base64_data.decode()],
                'req_id': task_id,
            }

            resp = self.client.post(URL, data=json.dumps(post_data))
            if resp.status_code == 200:
                print('request success')
            else:
                print('request failed')


class WebBehavior(locust.HttpUser):
    """
    simulate server
    """
    tasks = [ClientBehavior]
    min_wait = 10
    max_wait = 40
