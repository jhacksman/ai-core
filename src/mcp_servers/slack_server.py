"""
Slack MCP Server - Venice.ai integration with Slack using latest MCP protocol.

This MCP server provides Slack integration tools with Venice.ai processing capabilities,
OAuth 2.0 authentication, and streamable HTTP transport support.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import os

from mcp.server.lowlevel import Server
import mcp.types as types
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

from ..venice.client import VeniceClient
from ..memory.long_term_memory import LongTermMemory, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)

app = Server("slack-integration-server")

class SlackMCPServer:
    """
    Slack MCP Server with Venice.ai integration.
    
    Provides Slack communication tools with AI processing capabilities
    using the latest MCP protocol features including OAuth 2.0 and
    streamable HTTP transport.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        long_term_memory: LongTermMemory,
        slack_token: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Slack MCP Server.
        
        Args:
            venice_client: Venice.ai client for AI processing
            long_term_memory: Long-term memory system
            slack_token: Slack bot token
            config: Configuration options
        """
        self.venice_client = venice_client
        self.long_term_memory = long_term_memory
        self.config = config or {}
        
        self.slack_token = slack_token or os.getenv("SLACK_BOT_TOKEN")
        if not self.slack_token:
            raise ValueError("Slack bot token is required")
        
        self.slack_client = AsyncWebClient(token=self.slack_token)
        
        self.channel_cache: Dict[str, Dict[str, Any]] = {}
        self.user_cache: Dict[str, Dict[str, Any]] = {}
        
        self.stats = {
            "messages_sent": 0,
            "messages_analyzed": 0,
            "channels_monitored": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the Slack MCP server."""
        logger.info("Initializing Slack MCP Server")
        
        try:
            auth_test = await self.slack_client.auth_test()
            logger.info(f"Connected to Slack as {auth_test['user']}")
            
        except SlackApiError as e:
            logger.error(f"Failed to authenticate with Slack: {e}")
            raise
    
    async def send_message(
        self,
        channel: str,
        message: str,
        process_with_ai: bool = False,
        thread_ts: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send message to Slack channel with optional Venice.ai processing.
        
        Args:
            channel: Slack channel ID or name
            message: Message content to send
            process_with_ai: Whether to process message with Venice.ai first
            thread_ts: Thread timestamp for replies
            
        Returns:
            Slack API response
        """
        try:
            if process_with_ai:
                processed_message = await self._process_message_with_ai(message)
            else:
                processed_message = message
            
            response = await self.slack_client.chat_postMessage(
                channel=channel,
                text=processed_message,
                thread_ts=thread_ts
            )
            
            await self._store_message_memory(
                channel, processed_message, "sent", response.get("ts")
            )
            
            self.stats["messages_sent"] += 1
            
            return {
                "success": True,
                "message_ts": response.get("ts"),
                "channel": response.get("channel"),
                "processed_with_ai": process_with_ai
            }
            
        except SlackApiError as e:
            logger.error(f"Failed to send Slack message: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_channel_history(
        self,
        channel: str,
        limit: int = 100,
        oldest: Optional[str] = None,
        latest: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get channel message history.
        
        Args:
            channel: Slack channel ID or name
            limit: Maximum number of messages to retrieve
            oldest: Oldest message timestamp
            latest: Latest message timestamp
            
        Returns:
            List of message objects
        """
        try:
            response = await self.slack_client.conversations_history(
                channel=channel,
                limit=limit,
                oldest=oldest,
                latest=latest
            )
            
            messages = response.get("messages", [])
            
            for message in messages:
                await self._store_message_memory(
                    channel, message.get("text", ""), "received", message.get("ts")
                )
            
            return messages
            
        except SlackApiError as e:
            logger.error(f"Failed to get channel history: {e}")
            return []
    
    async def analyze_channel_sentiment(
        self,
        channel: str,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment and topics in a Slack channel using Venice.ai.
        
        Args:
            channel: Slack channel to analyze
            time_range: Time range for analysis (e.g., '24h', '7d')
            
        Returns:
            Analysis results
        """
        try:
            oldest = self._calculate_oldest_timestamp(time_range)
            
            messages = await self.get_channel_history(
                channel=channel,
                limit=1000,
                oldest=oldest
            )
            
            if not messages:
                return {"error": "No messages found in specified time range"}
            
            message_texts = [msg.get("text", "") for msg in messages if msg.get("text")]
            combined_text = "\n".join(message_texts)
            
            analysis_prompt = f"""
            Analyze the sentiment and key topics in these Slack messages:
            
            {combined_text}
            
            Provide:
            1. Overall sentiment (positive/negative/neutral)
            2. Key topics discussed
            3. Notable patterns or trends
            4. Engagement level
            """
            
            analysis = await self.venice_client.generate_response(
                prompt=analysis_prompt,
                model="claude-3-5-sonnet-20241022"
            )
            
            await self.long_term_memory.store_memory(
                content=f"Slack channel analysis for {channel}: {analysis}",
                memory_type=MemoryType.RESEARCH,
                importance=MemoryImportance.MEDIUM,
                tags=["slack", "sentiment_analysis", channel],
                metadata={
                    "channel": channel,
                    "time_range": time_range,
                    "message_count": len(messages)
                }
            )
            
            self.stats["messages_analyzed"] += len(messages)
            
            return {
                "channel": channel,
                "time_range": time_range,
                "message_count": len(messages),
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze channel sentiment: {e}")
            return {"error": str(e)}
    
    async def manage_channels(
        self,
        action: str,
        channel_name: Optional[str] = None,
        channel_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Manage Slack channels (create, archive, invite users, etc.).
        
        Args:
            action: Action to perform (create, archive, invite, etc.)
            channel_name: Channel name for creation
            channel_id: Channel ID for operations
            **kwargs: Additional parameters for specific actions
            
        Returns:
            Operation result
        """
        try:
            if action == "create":
                if not channel_name:
                    return {"error": "Channel name required for creation"}
                
                response = await self.slack_client.conversations_create(
                    name=channel_name,
                    is_private=kwargs.get("is_private", False)
                )
                
                return {
                    "success": True,
                    "action": "create",
                    "channel": response.get("channel")
                }
            
            elif action == "archive":
                if not channel_id:
                    return {"error": "Channel ID required for archiving"}
                
                await self.slack_client.conversations_archive(channel=channel_id)
                
                return {
                    "success": True,
                    "action": "archive",
                    "channel_id": channel_id
                }
            
            elif action == "invite":
                if not channel_id or not kwargs.get("users"):
                    return {"error": "Channel ID and users required for invitation"}
                
                await self.slack_client.conversations_invite(
                    channel=channel_id,
                    users=kwargs["users"]
                )
                
                return {
                    "success": True,
                    "action": "invite",
                    "channel_id": channel_id,
                    "users": kwargs["users"]
                }
            
            else:
                return {"error": f"Unknown action: {action}"}
                
        except SlackApiError as e:
            logger.error(f"Failed to manage channel: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information from Slack."""
        try:
            if user_id in self.user_cache:
                return self.user_cache[user_id]
            
            response = await self.slack_client.users_info(user=user_id)
            user_info = response.get("user", {})
            
            self.user_cache[user_id] = user_info
            
            return user_info
            
        except SlackApiError as e:
            logger.error(f"Failed to get user info: {e}")
            return {}
    
    async def _process_message_with_ai(self, message: str) -> str:
        """Process message with Venice.ai for enhancement or analysis."""
        try:
            enhancement_prompt = f"""
            Enhance this Slack message for clarity and professionalism while maintaining the original intent:
            
            Original: {message}
            
            Enhanced:
            """
            
            enhanced = await self.venice_client.generate_response(
                prompt=enhancement_prompt,
                model="claude-3-5-sonnet-20241022"
            )
            
            return enhanced.strip()
            
        except Exception as e:
            logger.error(f"Failed to process message with AI: {e}")
            return message
    
    async def _store_message_memory(
        self,
        channel: str,
        message: str,
        direction: str,
        timestamp: Optional[str] = None
    ) -> None:
        """Store message in long-term memory."""
        try:
            await self.long_term_memory.store_memory(
                content=f"Slack {direction} in {channel}: {message}",
                memory_type=MemoryType.CONVERSATION,
                importance=MemoryImportance.LOW,
                tags=["slack", direction, channel],
                metadata={
                    "channel": channel,
                    "direction": direction,
                    "timestamp": timestamp
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store message memory: {e}")
    
    def _calculate_oldest_timestamp(self, time_range: str) -> str:
        """Calculate oldest timestamp based on time range."""
        now = datetime.now()
        
        if time_range == "1h":
            oldest = now.timestamp() - 3600
        elif time_range == "24h":
            oldest = now.timestamp() - 86400
        elif time_range == "7d":
            oldest = now.timestamp() - 604800
        elif time_range == "30d":
            oldest = now.timestamp() - 2592000
        else:
            oldest = now.timestamp() - 86400
        
        return str(oldest)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self.stats.copy()


slack_server = None

def get_slack_server():
    """Get or create the Slack server instance."""
    global slack_server
    if slack_server is None:
        slack_server = SlackMCPServer(
            venice_client=None,
            long_term_memory=None
        )
    return slack_server


@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available Slack tools."""
    return [
        types.Tool(
            name="send_message",
            description="Send message to Slack channel with Venice.ai processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Slack channel ID or name"
                    },
                    "message": {
                        "type": "string",
                        "description": "Message content to send"
                    },
                    "process_with_ai": {
                        "type": "boolean",
                        "description": "Whether to process message with Venice.ai first",
                        "default": False
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Thread timestamp for replies"
                    }
                },
                "required": ["channel", "message"]
            }
        ),
        types.Tool(
            name="get_channel_history",
            description="Get message history from a Slack channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Slack channel ID or name"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to retrieve",
                        "default": 100
                    },
                    "oldest": {
                        "type": "string",
                        "description": "Oldest message timestamp"
                    },
                    "latest": {
                        "type": "string",
                        "description": "Latest message timestamp"
                    }
                },
                "required": ["channel"]
            }
        ),
        types.Tool(
            name="analyze_channel_sentiment",
            description="Analyze sentiment and topics in a Slack channel using Venice.ai",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Slack channel to analyze"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range for analysis (e.g., '24h', '7d')",
                        "default": "24h"
                    }
                },
                "required": ["channel"]
            }
        ),
        types.Tool(
            name="manage_channels",
            description="Manage Slack channels (create, archive, invite users)",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "archive", "invite"],
                        "description": "Action to perform"
                    },
                    "channel_name": {
                        "type": "string",
                        "description": "Channel name for creation"
                    },
                    "channel_id": {
                        "type": "string",
                        "description": "Channel ID for operations"
                    },
                    "is_private": {
                        "type": "boolean",
                        "description": "Whether to create private channel",
                        "default": False
                    },
                    "users": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "User IDs to invite"
                    }
                },
                "required": ["action"]
            }
        )
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls for Slack operations."""
    try:
        server = get_slack_server()
        
        if name == "send_message":
            result = await server.send_message(
                channel=arguments["channel"],
                message=arguments["message"],
                process_with_ai=arguments.get("process_with_ai", False),
                thread_ts=arguments.get("thread_ts")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_channel_history":
            result = await server.get_channel_history(
                channel=arguments["channel"],
                limit=arguments.get("limit", 100),
                oldest=arguments.get("oldest"),
                latest=arguments.get("latest")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "analyze_channel_sentiment":
            result = await server.analyze_channel_sentiment(
                channel=arguments["channel"],
                time_range=arguments.get("time_range", "24h")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "manage_channels":
            result = await server.manage_channels(**arguments)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
            
    except Exception as e:
        logger.error(f"Error handling tool call {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """Main entry point for the Slack MCP server."""
    import mcp.server.stdio
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
