#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time      :  2025/5/6 下午5:10
# @Author    :  MaybeShewill-CV
# @Site      :  ICode
# @Filename  :  mortred_vision_mcp_server.py
# @IDE:      :  PyCharm
import json
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn
import base64
import logging
from typing import (
    Any,
    TypeAlias,
)
from collections.abc import Callable
from functools import wraps

import utils

AnyFunction: TypeAlias = Callable[..., Any]

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # 设置日志格式
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server for Weather tools (SSE)
mcp = FastMCP("mortred_vision_server")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "mortred_vision_server/1.0"


async def make_nws_request(url: str, json) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, timeout=30.0, json=json)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.info(f"请求错误: {e}")
            return None


async def fetch_image_content(url: str) -> str:
    """

    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()  # 检查请求是否成功
            image_data = response.content
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            return base64_encoded
        except httpx.RequestError as e:
            logger.info(f"请求错误: {e}")
            return ""


def fetch_server_url(server_url):
    """

    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, _server_url=server_url, **kwargs)

        return wrapper

    return decorator


def add_mortred_vision_mcp_server():
    """

    """
    all_vision_servers = utils.scan_available_server()
    for server_info in all_vision_servers:
        name = server_info.model_name
        des = server_info.description
        url = server_info.url

        @mcp.tool(name=name, description=des)
        @fetch_server_url(server_url=url)
        async def run(image_url, **kwargs):
            b64_image = await fetch_image_content(image_url)
            logger.info('download image complete')
            if not b64_image:
                return b64_image
            json_data = {
                'img_data': b64_image,
                'req_id': 'test'
            }
            server_url = kwargs.get('_server_url')
            response = await make_nws_request(server_url, json_data)
            if response['code'] != 0:
                logger.info(f"请求错误, error_code: {response['data']}, error_msg: {response['msg']}")
                return response['msg']
            else:
                resp = json.dumps(response['data'])
                logger.info(f"status: {response['code']}, msg:{response['msg']}, resp: {resp}")
                return resp


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='Port to listen on')
    args = parser.parse_args()

    # add mortred vision mcp servers
    add_mortred_vision_mcp_server()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
