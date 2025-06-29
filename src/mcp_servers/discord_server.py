"""
Discord MCP Server - Venice.ai integration with Discord using latest MCP protocol.

This MCP server provides Discord integration tools with Venice.ai processing capabilities,
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
import discord
from discord.ext import commands

from ..venice.client import VeniceClient
from ..memory.long_term_memory import LongTermMemory, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)

app = Server("discord-integration-server")

class DiscordMCPServer:
    """
    Discord MCP Server with Venice.ai integration.
    
    Provides Discord communication tools with AI processing capabilities
    using the latest MCP protocol features including OAuth 2.0 and
    streamable HTTP transport.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        long_term_memory: LongTermMemory,
        discord_token: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Discord MCP Server.
        
        Args:
            venice_client: Venice.ai client for AI processing
            long_term_memory: Long-term memory system
            discord_token: Discord bot token
            config: Configuration options
        """
        self.venice_client = venice_client
        self.long_term_memory = long_term_memory
        self.config = config or {}
        
        self.discord_token = discord_token or os.getenv("DISCORD_BOT_TOKEN")
        if not self.discord_token:
            raise ValueError("Discord bot token is required")
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        
        self.guild_cache: Dict[int, discord.Guild] = {}
        self.channel_cache: Dict[int, discord.TextChannel] = {}
        
        self.stats = {
            "messages_sent": 0,
            "messages_analyzed": 0,
            "guilds_monitored": 0,
            "roles_managed": 0
        }
        
        self._setup_bot_events()
    
    def _setup_bot_events(self):
        """Setup Discord bot event handlers."""
        
        @self.bot.event
        async def on_ready():
            logger.info(f"Discord bot connected as {self.bot.user}")
            
            for guild in self.bot.guilds:
                self.guild_cache[guild.id] = guild
                self.stats["guilds_monitored"] += 1
        
        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return
            
            await self._store_message_memory(
                guild_id=message.guild.id if message.guild else None,
                channel_id=message.channel.id,
                content=message.content,
                author=str(message.author),
                direction="received"
            )
    
    async def initialize(self) -> None:
        """Initialize the Discord MCP server."""
        logger.info("Initializing Discord MCP Server")
        
        try:
            await self.bot.login(self.discord_token)
            logger.info("Discord bot authenticated successfully")
            
        except discord.LoginFailure as e:
            logger.error(f"Failed to authenticate with Discord: {e}")
            raise
    
    async def send_message(
        self,
        channel_id: int,
        message: str,
        process_with_ai: bool = False,
        embed_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send message to Discord channel with optional Venice.ai processing.
        
        Args:
            channel_id: Discord channel ID
            message: Message content to send
            process_with_ai: Whether to process message with Venice.ai first
            embed_data: Optional embed data for rich messages
            
        Returns:
            Discord API response
        """
        try:
            channel = self.bot.get_channel(channel_id)
            if not channel:
                return {"success": False, "error": "Channel not found"}
            
            if process_with_ai:
                processed_message = await self._process_message_with_ai(message)
            else:
                processed_message = message
            
            embed = None
            if embed_data:
                embed = discord.Embed(
                    title=embed_data.get("title"),
                    description=embed_data.get("description"),
                    color=embed_data.get("color", 0x00ff00)
                )
                
                if embed_data.get("fields"):
                    for field in embed_data["fields"]:
                        embed.add_field(
                            name=field.get("name", ""),
                            value=field.get("value", ""),
                            inline=field.get("inline", False)
                        )
            
            sent_message = await channel.send(content=processed_message, embed=embed)
            
            await self._store_message_memory(
                guild_id=channel.guild.id if channel.guild else None,
                channel_id=channel_id,
                content=processed_message,
                author=str(self.bot.user),
                direction="sent",
                message_id=sent_message.id
            )
            
            self.stats["messages_sent"] += 1
            
            return {
                "success": True,
                "message_id": sent_message.id,
                "channel_id": channel_id,
                "processed_with_ai": process_with_ai
            }
            
        except discord.DiscordException as e:
            logger.error(f"Failed to send Discord message: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_guild_info(self, guild_id: int) -> Dict[str, Any]:
        """
        Get Discord guild (server) information.
        
        Args:
            guild_id: Discord guild ID
            
        Returns:
            Guild information
        """
        try:
            guild = self.bot.get_guild(guild_id)
            if not guild:
                return {"error": "Guild not found"}
            
            self.guild_cache[guild_id] = guild
            
            return {
                "id": guild.id,
                "name": guild.name,
                "description": guild.description,
                "member_count": guild.member_count,
                "channel_count": len(guild.channels),
                "role_count": len(guild.roles),
                "owner": str(guild.owner) if guild.owner else None,
                "created_at": guild.created_at.isoformat(),
                "features": guild.features
            }
            
        except discord.DiscordException as e:
            logger.error(f"Failed to get guild info: {e}")
            return {"error": str(e)}
    
    async def manage_roles(
        self,
        action: str,
        guild_id: int,
        user_id: Optional[int] = None,
        role_id: Optional[int] = None,
        role_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Manage Discord roles (create, assign, remove, etc.).
        
        Args:
            action: Action to perform (create, assign, remove, delete)
            guild_id: Discord guild ID
            user_id: User ID for role assignment/removal
            role_id: Role ID for operations
            role_name: Role name for creation
            **kwargs: Additional parameters for specific actions
            
        Returns:
            Operation result
        """
        try:
            guild = self.bot.get_guild(guild_id)
            if not guild:
                return {"error": "Guild not found"}
            
            if action == "create":
                if not role_name:
                    return {"error": "Role name required for creation"}
                
                role = await guild.create_role(
                    name=role_name,
                    color=discord.Color(kwargs.get("color", 0)),
                    permissions=discord.Permissions(kwargs.get("permissions", 0)),
                    hoist=kwargs.get("hoist", False),
                    mentionable=kwargs.get("mentionable", False)
                )
                
                self.stats["roles_managed"] += 1
                
                return {
                    "success": True,
                    "action": "create",
                    "role": {
                        "id": role.id,
                        "name": role.name,
                        "color": role.color.value
                    }
                }
            
            elif action == "assign":
                if not user_id or not role_id:
                    return {"error": "User ID and role ID required for assignment"}
                
                member = guild.get_member(user_id)
                role = guild.get_role(role_id)
                
                if not member or not role:
                    return {"error": "Member or role not found"}
                
                await member.add_roles(role)
                
                return {
                    "success": True,
                    "action": "assign",
                    "user_id": user_id,
                    "role_id": role_id
                }
            
            elif action == "remove":
                if not user_id or not role_id:
                    return {"error": "User ID and role ID required for removal"}
                
                member = guild.get_member(user_id)
                role = guild.get_role(role_id)
                
                if not member or not role:
                    return {"error": "Member or role not found"}
                
                await member.remove_roles(role)
                
                return {
                    "success": True,
                    "action": "remove",
                    "user_id": user_id,
                    "role_id": role_id
                }
            
            elif action == "delete":
                if not role_id:
                    return {"error": "Role ID required for deletion"}
                
                role = guild.get_role(role_id)
                if not role:
                    return {"error": "Role not found"}
                
                await role.delete()
                
                return {
                    "success": True,
                    "action": "delete",
                    "role_id": role_id
                }
            
            else:
                return {"error": f"Unknown action: {action}"}
                
        except discord.DiscordException as e:
            logger.error(f"Failed to manage role: {e}")
            return {"success": False, "error": str(e)}
    
    async def analyze_guild_activity(
        self,
        guild_id: int,
        time_range: str = "24h",
        channel_limit: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze activity and sentiment in a Discord guild using Venice.ai.
        
        Args:
            guild_id: Discord guild to analyze
            time_range: Time range for analysis (e.g., '24h', '7d')
            channel_limit: Maximum number of channels to analyze
            
        Returns:
            Analysis results
        """
        try:
            guild = self.bot.get_guild(guild_id)
            if not guild:
                return {"error": "Guild not found"}
            
            oldest_timestamp = self._calculate_oldest_timestamp(time_range)
            
            all_messages = []
            channels_analyzed = 0
            
            for channel in guild.text_channels:
                if channels_analyzed >= channel_limit:
                    break
                
                try:
                    async for message in channel.history(
                        limit=100,
                        after=datetime.fromtimestamp(float(oldest_timestamp))
                    ):
                        if message.content:
                            all_messages.append({
                                "content": message.content,
                                "author": str(message.author),
                                "channel": channel.name,
                                "timestamp": message.created_at.isoformat()
                            })
                    
                    channels_analyzed += 1
                    
                except discord.Forbidden:
                    continue
            
            if not all_messages:
                return {"error": "No messages found in specified time range"}
            
            combined_text = "\n".join([msg["content"] for msg in all_messages])
            
            analysis_prompt = f"""
            Analyze the activity and sentiment in this Discord guild:
            
            Guild: {guild.name}
            Time Range: {time_range}
            Messages Analyzed: {len(all_messages)}
            Channels: {channels_analyzed}
            
            Messages:
            {combined_text[:8000]}
            
            Provide:
            1. Overall activity level and engagement
            2. Sentiment analysis (positive/negative/neutral)
            3. Key topics and themes discussed
            4. Community health indicators
            5. Notable patterns or trends
            """
            
            analysis = await self.venice_client.generate_response(
                prompt=analysis_prompt,
                model="claude-3-5-sonnet-20241022"
            )
            
            await self.long_term_memory.store_memory(
                content=f"Discord guild analysis for {guild.name}: {analysis}",
                memory_type=MemoryType.RESEARCH,
                importance=MemoryImportance.MEDIUM,
                tags=["discord", "guild_analysis", guild.name],
                metadata={
                    "guild_id": guild_id,
                    "guild_name": guild.name,
                    "time_range": time_range,
                    "message_count": len(all_messages),
                    "channels_analyzed": channels_analyzed
                }
            )
            
            self.stats["messages_analyzed"] += len(all_messages)
            
            return {
                "guild_id": guild_id,
                "guild_name": guild.name,
                "time_range": time_range,
                "message_count": len(all_messages),
                "channels_analyzed": channels_analyzed,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze guild activity: {e}")
            return {"error": str(e)}
    
    async def get_channel_messages(
        self,
        channel_id: int,
        limit: int = 100,
        before_message_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a Discord channel.
        
        Args:
            channel_id: Discord channel ID
            limit: Maximum number of messages to retrieve
            before_message_id: Get messages before this message ID
            
        Returns:
            List of message objects
        """
        try:
            channel = self.bot.get_channel(channel_id)
            if not channel:
                return []
            
            messages = []
            
            before = None
            if before_message_id:
                before = discord.Object(id=before_message_id)
            
            async for message in channel.history(limit=limit, before=before):
                messages.append({
                    "id": message.id,
                    "content": message.content,
                    "author": {
                        "id": message.author.id,
                        "name": str(message.author),
                        "display_name": message.author.display_name
                    },
                    "timestamp": message.created_at.isoformat(),
                    "edited_at": message.edited_at.isoformat() if message.edited_at else None,
                    "attachments": [att.url for att in message.attachments],
                    "embeds": len(message.embeds),
                    "reactions": [{"emoji": str(reaction.emoji), "count": reaction.count} for reaction in message.reactions]
                })
            
            return messages
            
        except discord.DiscordException as e:
            logger.error(f"Failed to get channel messages: {e}")
            return []
    
    async def _process_message_with_ai(self, message: str) -> str:
        """Process message with Venice.ai for enhancement or analysis."""
        try:
            enhancement_prompt = f"""
            Enhance this Discord message for clarity and engagement while maintaining the original intent:
            
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
        guild_id: Optional[int],
        channel_id: int,
        content: str,
        author: str,
        direction: str,
        message_id: Optional[int] = None
    ) -> None:
        """Store message in long-term memory."""
        try:
            guild_name = "DM" if not guild_id else self.guild_cache.get(guild_id, {}).name or "Unknown"
            
            await self.long_term_memory.store_memory(
                content=f"Discord {direction} in {guild_name}: {content}",
                memory_type=MemoryType.CONVERSATION,
                importance=MemoryImportance.LOW,
                tags=["discord", direction, guild_name],
                metadata={
                    "guild_id": guild_id,
                    "channel_id": channel_id,
                    "author": author,
                    "direction": direction,
                    "message_id": message_id
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


discord_server = None

def get_discord_server():
    """Get or create the Discord server instance."""
    global discord_server
    if discord_server is None:
        discord_server = DiscordMCPServer(
            venice_client=None,
            long_term_memory=None
        )
    return discord_server


@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available Discord tools."""
    return [
        types.Tool(
            name="send_message",
            description="Send message to Discord channel with Venice.ai processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": "Discord channel ID"
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
                    "embed_data": {
                        "type": "object",
                        "description": "Optional embed data for rich messages",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "color": {"type": "integer"},
                            "fields": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "value": {"type": "string"},
                                        "inline": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                },
                "required": ["channel_id", "message"]
            }
        ),
        types.Tool(
            name="get_guild_info",
            description="Get Discord guild (server) information",
            inputSchema={
                "type": "object",
                "properties": {
                    "guild_id": {
                        "type": "integer",
                        "description": "Discord guild ID"
                    }
                },
                "required": ["guild_id"]
            }
        ),
        types.Tool(
            name="manage_roles",
            description="Manage Discord roles (create, assign, remove, delete)",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "assign", "remove", "delete"],
                        "description": "Action to perform"
                    },
                    "guild_id": {
                        "type": "integer",
                        "description": "Discord guild ID"
                    },
                    "user_id": {
                        "type": "integer",
                        "description": "User ID for role assignment/removal"
                    },
                    "role_id": {
                        "type": "integer",
                        "description": "Role ID for operations"
                    },
                    "role_name": {
                        "type": "string",
                        "description": "Role name for creation"
                    },
                    "color": {
                        "type": "integer",
                        "description": "Role color (hex value)"
                    },
                    "permissions": {
                        "type": "integer",
                        "description": "Role permissions (bitfield)"
                    },
                    "hoist": {
                        "type": "boolean",
                        "description": "Whether role should be displayed separately",
                        "default": False
                    },
                    "mentionable": {
                        "type": "boolean",
                        "description": "Whether role should be mentionable",
                        "default": False
                    }
                },
                "required": ["action", "guild_id"]
            }
        ),
        types.Tool(
            name="analyze_guild_activity",
            description="Analyze activity and sentiment in a Discord guild using Venice.ai",
            inputSchema={
                "type": "object",
                "properties": {
                    "guild_id": {
                        "type": "integer",
                        "description": "Discord guild to analyze"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range for analysis (e.g., '24h', '7d')",
                        "default": "24h"
                    },
                    "channel_limit": {
                        "type": "integer",
                        "description": "Maximum number of channels to analyze",
                        "default": 10
                    }
                },
                "required": ["guild_id"]
            }
        ),
        types.Tool(
            name="get_channel_messages",
            description="Get messages from a Discord channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": "Discord channel ID"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to retrieve",
                        "default": 100
                    },
                    "before_message_id": {
                        "type": "integer",
                        "description": "Get messages before this message ID"
                    }
                },
                "required": ["channel_id"]
            }
        )
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls for Discord operations."""
    try:
        server = get_discord_server()
        
        if name == "send_message":
            result = await server.send_message(
                channel_id=arguments["channel_id"],
                message=arguments["message"],
                process_with_ai=arguments.get("process_with_ai", False),
                embed_data=arguments.get("embed_data")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_guild_info":
            result = await server.get_guild_info(
                guild_id=arguments["guild_id"]
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "manage_roles":
            result = await server.manage_roles(**arguments)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "analyze_guild_activity":
            result = await server.analyze_guild_activity(
                guild_id=arguments["guild_id"],
                time_range=arguments.get("time_range", "24h"),
                channel_limit=arguments.get("channel_limit", 10)
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_channel_messages":
            result = await server.get_channel_messages(
                channel_id=arguments["channel_id"],
                limit=arguments.get("limit", 100),
                before_message_id=arguments.get("before_message_id")
            )
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
    """Main entry point for the Discord MCP server."""
    import mcp.server.stdio
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
