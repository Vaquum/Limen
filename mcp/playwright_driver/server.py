#!/usr/bin/env python3
import asyncio
from typing import Optional, Dict, Any, List

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from playwright.async_api import async_playwright


class PW:
    browser = None
    page = None
    ctx = None
    _pl = None


pw = PW()
server = Server('playwright-driver')


@server.list_tools()
async def list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name='open',
            description='Launch Chromium and open a new page at url',
            inputSchema={'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']},
        ),
        types.Tool(
            name='goto',
            description='Navigate current page to url',
            inputSchema={'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']},
        ),
        types.Tool(
            name='click',
            description='Click a CSS selector',
            inputSchema={'type': 'object', 'properties': {'selector': {'type': 'string'}}, 'required': ['selector']},
        ),
        types.Tool(
            name='type',
            description='Type text into a CSS selector (focusable/input)',
            inputSchema={
                'type': 'object',
                'properties': {'selector': {'type': 'string'}, 'text': {'type': 'string'}},
                'required': ['selector', 'text'],
            },
        ),
        types.Tool(
            name='wait_for_selector',
            description='Wait for selector to appear',
            inputSchema={
                'type': 'object',
                'properties': {'selector': {'type': 'string'}, 'timeout': {'type': 'integer'}},
                'required': ['selector'],
            },
        ),
        types.Tool(
            name='screenshot',
            description='Return a PNG screenshot of the current page',
            inputSchema={'type': 'object', 'properties': {'fullPage': {'type': 'boolean'}}},
        ),
        types.Tool(
            name='close',
            description='Close browser',
            inputSchema={'type': 'object', 'properties': {}},
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, args: Optional[Dict[str, Any]]
) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    args = args or {}

    async def ensure_browser():
        if pw.browser is None:
            pw._pl = await async_playwright().start()
            pw.browser = await pw._pl.chromium.launch(headless=True)
            pw.ctx = await pw.browser.new_context()
            pw.page = await pw.ctx.new_page()

    if name == 'open':
        await ensure_browser()
        await pw.page.goto(args['url'])
        return [types.TextContent(type='text', text='ok: opened')]

    if name == 'goto':
        await ensure_browser()
        await pw.page.goto(args['url'])
        return [types.TextContent(type='text', text='ok: navigated')]

    if name == 'click':
        await ensure_browser()
        await pw.page.click(args['selector'])
        return [types.TextContent(type='text', text='ok: clicked')]

    if name == 'type':
        await ensure_browser()
        await pw.page.fill(args['selector'], args['text'])
        return [types.TextContent(type='text', text='ok: typed')]

    if name == 'wait_for_selector':
        await ensure_browser()
        timeout = int(args.get('timeout', 15000))
        await pw.page.wait_for_selector(args['selector'], timeout=timeout)
        return [types.TextContent(type='text', text='ok: seen')]

    if name == 'screenshot':
        await ensure_browser()
        img = await pw.page.screenshot(full_page=bool(args.get('fullPage', False)))
        return [types.ImageContent(type='image', data=img, mimeType='image/png')]

    if name == 'close':
        if pw.browser:
            await pw.browser.close()
        if pw._pl:
            await pw._pl.stop()
        pw.browser = pw.page = pw.ctx = pw._pl = None
        return [types.TextContent(type='text', text='ok: closed')]

    return [types.TextContent(type='text', text=f'unknown tool: {name}')]


async def main():
    async with mcp.server.stdio.stdio_server() as (r, w):
        await server.run(
            r,
            w,
            InitializationOptions(
                server_name='playwright-driver',
                server_version='1.0.0',
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == '__main__':
    asyncio.run(main())