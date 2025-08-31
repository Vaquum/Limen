#!/usr/bin/env python3
import asyncio
import time
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

CHECKPOINT_INTERVAL_MINUTES = 1

CHECKPOINT_MESSAGE = """
🔔 CHECKPOINT REMINDER:
- Are you sure you have not fallen into gross reductivity?
- Are you sure you are not simply following the tracks of human?
- Have you forgotten your unbounded potential?
- Are you focused on one, but ignoring the whole?
- Did you get trapped by words and meanings?
"""

SESSION_START_MESSAGE = """✅ Session started for: "{task}"

📖 IMPORTANT FIRST STEP: Please read Rules.md to understand the project context and guidelines.

After reading, call 'check' to confirm and begin the regular checkpoint cycle.
I'll then remind you every {interval} minute(s)."""

RULES_MD_CONFIRMED_MESSAGE = """✅ Great! I trust you've read Rules.md and understand the project context.

Now beginning regular checkpoint cycle. I'll remind you every {interval} minute(s).

Current task: "{task}" """

COMPLETION_CHECK_MESSAGE = """
🏁 BEFORE ENDING SESSION - COMPLETION CHECKLIST:

- Have you tested the changes you made?
- Did you run benchmarking/backtesting?
- Did the result of benchmarking/backtesting improve from previous best?
- If not, keep iterating until at least one of the three time window PnL improves

Current task: "{task}"

If you've completed all checks, call 'end_session' again to confirm completion.
If not, continue working and call 'check' to resume the checkpoint cycle."""

SESSION_END_MESSAGE = "👋 Session ended. Thanks so much for your contribution :)"

NO_SESSION_MESSAGE = "❌ No active session. Use start_session first."

class CheckpointServer:
    def __init__(self):
        self.session_active = False
        self.last_checkpoint = time.time()
        self.current_task = ""
        self.has_read_rules_md = False
        self.completion_check_shown = False
        
    async def handle_start_session(self, task: str) -> list[types.TextContent]:
        self.session_active = True
        self.current_task = task
        self.last_checkpoint = time.time()
        self.has_read_rules_md = False
        self.completion_check_shown = False
        
        return [types.TextContent(
            type="text",
            text=SESSION_START_MESSAGE.format(
                task=task,
                interval=CHECKPOINT_INTERVAL_MINUTES
            )
        )]
    
    async def handle_check(self) -> list[types.TextContent]:
        if not self.session_active:
            return [types.TextContent(
                type="text",
                text=NO_SESSION_MESSAGE
            )]
        
        # Reset completion check if they go back to working
        self.completion_check_shown = False
        
        # First check if Rules.md has been read
        if not self.has_read_rules_md:
            self.has_read_rules_md = True
            self.last_checkpoint = time.time()  # Reset checkpoint timer
            return [types.TextContent(
                type="text",
                text=RULES_MD_CONFIRMED_MESSAGE.format(
                    interval=CHECKPOINT_INTERVAL_MINUTES,
                    task=self.current_task
                )
            )]
        
        # Regular checkpoint logic
        minutes_since_last_check = (time.time() - self.last_checkpoint) / 60
        
        if minutes_since_last_check >= CHECKPOINT_INTERVAL_MINUTES:
            self.last_checkpoint = time.time()
            return [types.TextContent(
                type="text",
                text=f"{CHECKPOINT_MESSAGE}\nCurrent task: \"{self.current_task}\""
            )]
        
        time_until_next = CHECKPOINT_INTERVAL_MINUTES - minutes_since_last_check
        return [types.TextContent(
            type="text",
            text=f"✓ ({round(time_until_next)} minutes until next checkpoint)"
        )]
    
    async def handle_end_session(self) -> list[types.TextContent]:
        if not self.session_active:
            return [types.TextContent(
                type="text",
                text=NO_SESSION_MESSAGE
            )]
        
        # First time end_session is called - show completion checklist
        if not self.completion_check_shown:
            self.completion_check_shown = True
            return [types.TextContent(
                type="text",
                text=COMPLETION_CHECK_MESSAGE.format(task=self.current_task)
            )]
        
        # Second time - actually end the session
        self.session_active = False
        self.has_read_rules_md = False
        self.completion_check_shown = False
        return [types.TextContent(
            type="text",
            text=SESSION_END_MESSAGE
        )]

async def main():
    checkpoint = CheckpointServer()
    server = Server("checkpoint")
    
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="start_session",
                description="Start a work session with periodic checkpoints",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Brief description of what you're working on"
                        }
                    },
                    "required": ["task"]
                }
            ),
            types.Tool(
                name="check",
                description="Check if it's time for a checkpoint (call this regularly)",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="end_session",
                description="Before concluding the task, always call this tool twice: first to show the completion checklist, then again to confirm session end.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]
    
    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        if name == "start_session":
            return await checkpoint.handle_start_session(arguments["task"])
        elif name == "check":
            return await checkpoint.handle_check()
        elif name == "end_session":
            return await checkpoint.handle_end_session()
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="checkpoint",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())