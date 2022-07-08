#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 22-7-2 上午1:12
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/MMAiServer
# @File    : test_server.py
# @IDE: PyCharm
"""
test server
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

from config_utils import parse_config_utils

CFG_MAP = parse_config_utils.cfg_map


def init_args():
    """
    int args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, help='model name')
    parser.add_argument('--mode', type=str, help='test mode')

    return parser.parse_args()


def single_test_mode(url, src_image_path, loop_times):
    """_summary_

    Args:
        url (_type_): _description_
        src_image_path (_type_): _description_
        loop_times (_type_): _description_
    """
    assert ops.exists(src_image_path), '{:s} not exist'.format(src_image_path)
    with open(src_image_path, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data)

    task_id = src_image_path + str(time.time())
    m2 = hashlib.md5()
    m2.update(task_id.encode())
    task_id = m2.hexdigest()
    post_data = {
        'img_data': base64_data.decode(),
        'req_id': task_id,
    }

    for i in range(loop_times):
        try:
            resp = requests.post(url=url, data=json.dumps(post_data))
            print(resp.text[:200])
        except Exception as e:
            print(e)
    return


def locust_test_mode(url, src_image_path, u, r, t):
    """_summary_

    Args:
        url (_type_): _description_
        src_image_path (_type_): _description_
        u (_type_): _description_
        r (_type_): _description_
        t (_type_): _description_
    """
    content = open('./server/locust_performance.py', 'r').readlines()
    content[17] = 'URL = \'{:s}\'\n'.format(url)
    content[18] = 'SRC_IMAGE_PATH = \'{:s}\'\n'.format(src_image_path)
    open('./server/locust_performance.py', 'w').writelines(content)

    command = 'locust -f ./server/locust_performance.py --host={:s} --headless -u {:d} -r {:d} -t {:s}'.format(url, u, r, t)
    os.system(command=command)

    return


def main_process():
    """
    main func
    """
    args = init_args()

    server_name = args.server.lower()
    if server_name not in CFG_MAP:
        print('No valid configuration for model: {:s}'.format(server_name))
        print('Supported servers are listed: ')
        print(CFG_MAP.keys())
        return
    test_mode = args.mode
    if test_mode not in ['single', 'locust']:
        print('Only support \'single\' and \'locust\' mode')
        return
    
    cfg = CFG_MAP[server_name]
    model_name = cfg.MODEL_NAME
    url = cfg.URL
    source_image_path = cfg.SOURCE_IMAGE_PATH
    if test_mode == 'single':
        loop_times = cfg.SINGLE.LOOP_TIMES
        print('Start test server for model: {:s}, mode {:s}'.format(model_name, test_mode))
        single_test_mode(
            url=url,
            src_image_path=source_image_path,
            loop_times=loop_times
        )
    else:
        u = cfg.LOCUST.U
        r = cfg.LOCUST.R
        t = cfg.LOCUST.T
        print('Start test server for model: {:s}, mode {:s}'.format(model_name, test_mode))
        locust_test_mode(
            url=url,
            src_image_path=source_image_path,
            u=u,
            r=r,
            t=t
        )
        content = open('./server/locust_performance.py', 'r').readlines()
        content[17] = 'URL = \'\'\n'
        content[18] = 'SRC_IMAGE_PATH = \'\'\n'
        open('./server/locust_performance.py', 'w').writelines(content)
    
    print('Complete test')
    return


if __name__ == '__main__':
    """
    main func
    """
    main_process()
