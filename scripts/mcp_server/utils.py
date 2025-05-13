#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time      :  2025/5/8 下午8:45
# @Author    :  MaybeShewill-CV
# @Site      :  ICode
# @Filename  :  utils.py
# @IDE:      :  PyCharm
import os
import os.path as ops
import json
import logging
import copy

import requests
import pprint


class VisionServerInfo(object):
    """

    """
    def __init__(self, url='', model_name='', description='', annotations=''):
        """

        :param url:
        :param model_name:
        :param description:
        :param annotations:
        """
        self._url = url
        self._model_name = model_name
        self._description = description
        self._annotations = annotations

    @property
    def url(self):
        """

        :return:
        """
        return self._url

    @property
    def model_name(self):
        """

        :return:
        """
        return self._model_name

    @property
    def description(self):
        """

        :return:
        """
        return self._description

    @property
    def annotations(self):
        """

        :return:
        """
        return self._annotations

    @url.setter
    def url(self, url):
        """

        """
        self._url = url

    @model_name.setter
    def model_name(self, model_name):
        """

        """
        self._model_name = model_name

    @description.setter
    def description(self, description):
        """

        """
        self._description = description

    @annotations.setter
    def annotations(self, annotations):
        """

        """
        self._annotations = annotations

    def info(self):
        """

        """
        info_dict = {
            'model_name': self._model_name,
            'description': self._description,
            'url': self._url,
            'annotation': self._annotations
        }

        return info_dict


def _send_heartbeat(url):
    """

    :param url:
    :return:
    """
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            # print("[OK] Heartbeat:", resp.text)
            return True
        else:
            # print("[ERROR] Heartbeat failed:", resp.status_code)
            return False
    except Exception as e:
        # print("[EXCEPTION] Heartbeat:", str(e))
        return False


def _parse_vision_server_info(conf_file_path):
    """

    :param conf_file_path:
    :return:
    """
    if not ops.exists(conf_file_path):
        print(f'{conf_file_path} not exist')
        return None

    server_conf = {}
    current_section = None

    with open(conf_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                server_conf[section] = {}
                current_section = section
            elif "=" in line and current_section:
                key, value = line.split("=", 1)
                server_conf[current_section][key.strip()] = value.strip().strip('"')

    port = None
    host = None
    server_uri = None
    for section, cfg in server_conf.items():
        if 'port' in cfg:
            port = cfg['port']
        if 'host' in cfg:
            host = cfg['host']
        if 'server_uri' in cfg:
            server_uri = cfg['server_uri']
    if port is None or host is None or server_uri is None:
        return None

    ret = {
        'server_url': f'http://{host}:{port}{server_uri}'
    }

    return ret


def _write_local_vision_mcp_server_cfg_file():
    """

    """
    description_map = {
        'classification': '按照imagenet预设的1000个类别，对图片进行分类',
        'enhancement': '对图像做质量增强',
        'feature_point': '检测图像上的特征点位置及其特征向量',
        'matting': '从图像中精确地分离前景与背景',
        'mono_depth_estimation': '使用单张图像估计图像上每个像素位置的深度',
        'object_detection': '按照coco数据集定义的80种类别，检测图像上目标物体的位置和类别',
        'scene_segmentation': '按照cityscapes数据集定义的类别，对图像上的每一个像素进行分类',
    }

    template_info = {
        "name": "",
        "description": "",
        "conf_path": "",
        "inputSchema": {
          "type": "object",
          "properties": {
            "req_id": {
              "type": "string",
              "description": "全局唯一的任务id"
            },
            "img_data": {
              "type": "string",
              "description": "按照base64编码的图像内容"
            }
          },
          "required": [
            "req_id",
            "img_data"
          ]
        },
        "annotations": {
          "title": "",
          "readOnlyHint": False,
          "destructiveHint": False,
          "idempotentHint": True,
          "openWorldHint": True
        }
    }

    all_server_infos = []

    base_dir = '../../conf/server'
    for dir_name in os.listdir(base_dir):
        if dir_name not in description_map:
            continue
        description = description_map[dir_name]
        dir_path = ops.join(base_dir, dir_name)
        if not ops.isdir(dir_path):
            continue
        for name in os.listdir(dir_path):
            server_info = copy.deepcopy(template_info)
            cfg_dir_path = ops.join(dir_path, name)
            if not ops.isdir(cfg_dir_path):
                continue
            server_info['description'] = description
            server_info['name'] = name
            for cfg_file_name in os.listdir(cfg_dir_path):
                cfg_file_path = ops.join(cfg_dir_path, cfg_file_name)
                if ops.isfile(cfg_file_path):
                    server_info['conf_path'] = cfg_file_path
                    break
            all_server_infos.append(server_info)

    with open('mortred_vision_server_cfg.json', 'w') as o_file:
        json.dump(all_server_infos, o_file, indent=2, ensure_ascii=False)


def scan_available_server():
    """

    """
    local_vision_mcp_cfg_path = './mortred_vision_server_cfg.json'
    available_server = []

    if not ops.exists(local_vision_mcp_cfg_path):
        raise ValueError(f'{local_vision_mcp_cfg_path} not exists')

    with open(local_vision_mcp_cfg_path, 'r', encoding='utf-8') as file:
        info = json.load(file)
        for model_info in info:
            vision_server = VisionServerInfo()
            vision_server.model_name = model_info['name']
            vision_server.description = model_info['description']
            cfg_file_path = model_info['conf_path']
            cfg_info = _parse_vision_server_info(cfg_file_path)
            if cfg_info is None:
                continue
            vision_server.url = cfg_info['server_url']
            vision_server.annotations = ''

            # send heartbeat test
            if not _send_heartbeat(vision_server.url):
                # print(f'vision server: {vision_server.url} not available')
                continue
            else:
                available_server.append(vision_server)
                print(f'****************** \n available vision server:\n  {vision_server.info()} ')

    return available_server


if __name__ == '__main__':
    """
    main func
    """
    # _write_local_vision_mcp_server_cfg_file()

    scan_available_server()
