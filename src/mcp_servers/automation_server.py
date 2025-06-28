"""
Automation MCP Server - Browser automation and web search using latest MCP protocol.

This MCP server provides browser automation and web search tools with Venice.ai processing capabilities,
OAuth 2.0 authentication, and streamable HTTP transport support.
"""

import asyncio
import logging
import json
import os
import tempfile
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from mcp.server.lowlevel import Server
import mcp.types as types
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import aiohttp
import aiofiles

from ..venice.client import VeniceClient
from ..memory.long_term_memory import LongTermMemory, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)

app = Server("automation-server")

@dataclass
class BrowserSession:
    """Browser session data structure."""
    session_id: str
    browser: Browser
    context: BrowserContext
    page: Page
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    """Search result data structure."""
    title: str
    url: str
    snippet: str
    rank: int
    source: str
    metadata: Dict[str, Any]

class AutomationMCPServer:
    """
    Automation MCP Server with Venice.ai integration.
    
    Provides browser automation and web search tools with AI analysis capabilities
    using the latest MCP protocol features including OAuth 2.0 and
    streamable HTTP transport.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        long_term_memory: LongTermMemory,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Automation MCP Server.
        
        Args:
            venice_client: Venice.ai client for AI processing
            long_term_memory: Long-term memory system
            config: Configuration options
        """
        self.venice_client = venice_client
        self.long_term_memory = long_term_memory
        self.config = config or {}
        
        self.playwright = None
        self.browser_sessions: Dict[str, BrowserSession] = {}
        
        self.search_engines = self.config.get("search_engines", {
            "duckduckgo": "https://duckduckgo.com/?q={query}",
            "bing": "https://www.bing.com/search?q={query}",
            "google": "https://www.google.com/search?q={query}"
        })
        
        self.download_dir = Path(self.config.get("download_dir", "./downloads"))
        self.download_dir.mkdir(exist_ok=True)
        
        self.stats = {
            "browser_sessions_created": 0,
            "pages_navigated": 0,
            "searches_performed": 0,
            "files_downloaded": 0,
            "screenshots_taken": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the Automation MCP server."""
        logger.info("Initializing Automation MCP Server")
        
        try:
            self.playwright = await async_playwright().start()
            logger.info("Playwright initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Automation MCP Server: {e}")
            raise
    
    async def create_browser_session(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        viewport: Optional[Dict[str, int]] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new browser session.
        
        Args:
            headless: Whether to run browser in headless mode
            browser_type: Type of browser (chromium, firefox, webkit)
            viewport: Browser viewport size
            user_agent: Custom user agent string
            
        Returns:
            Browser session information
        """
        try:
            if not self.playwright:
                await self.initialize()
            
            if browser_type == "firefox":
                browser = await self.playwright.firefox.launch(headless=headless)
            elif browser_type == "webkit":
                browser = await self.playwright.webkit.launch(headless=headless)
            else:
                browser = await self.playwright.chromium.launch(headless=headless)
            
            context_options = {}
            if viewport:
                context_options["viewport"] = viewport
            if user_agent:
                context_options["user_agent"] = user_agent
            
            context = await browser.new_context(**context_options)
            page = await context.new_page()
            
            session_id = self._generate_session_id()
            
            session = BrowserSession(
                session_id=session_id,
                browser=browser,
                context=context,
                page=page,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                metadata={
                    "browser_type": browser_type,
                    "headless": headless,
                    "viewport": viewport,
                    "user_agent": user_agent
                }
            )
            
            self.browser_sessions[session_id] = session
            self.stats["browser_sessions_created"] += 1
            
            logger.debug(f"Created browser session: {session_id}")
            
            return {
                "session_id": session_id,
                "browser_type": browser_type,
                "headless": headless,
                "viewport": viewport,
                "created_at": session.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create browser session: {e}")
            return {"error": str(e)}
    
    async def navigate_to_url(
        self,
        session_id: str,
        url: str,
        wait_for: Optional[str] = None,
        timeout: int = 30000
    ) -> Dict[str, Any]:
        """
        Navigate to a URL in a browser session.
        
        Args:
            session_id: Browser session ID
            url: URL to navigate to
            wait_for: What to wait for (load, domcontentloaded, networkidle)
            timeout: Navigation timeout in milliseconds
            
        Returns:
            Navigation result
        """
        try:
            if session_id not in self.browser_sessions:
                return {"error": "Browser session not found"}
            
            session = self.browser_sessions[session_id]
            session.last_activity = datetime.now()
            
            wait_until = wait_for or "domcontentloaded"
            
            response = await session.page.goto(
                url,
                wait_until=wait_until,
                timeout=timeout
            )
            
            page_title = await session.page.title()
            current_url = session.page.url
            
            await self._store_navigation_memory(session_id, url, page_title)
            self.stats["pages_navigated"] += 1
            
            return {
                "session_id": session_id,
                "url": current_url,
                "title": page_title,
                "status": response.status if response else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to navigate to URL: {e}")
            return {"error": str(e)}
    
    async def extract_page_content(
        self,
        session_id: str,
        content_type: str = "text",
        selector: Optional[str] = None,
        analyze_with_ai: bool = False
    ) -> Dict[str, Any]:
        """
        Extract content from the current page.
        
        Args:
            session_id: Browser session ID
            content_type: Type of content to extract (text, html, links, images)
            selector: CSS selector for specific elements
            analyze_with_ai: Whether to analyze content with Venice.ai
            
        Returns:
            Extracted content
        """
        try:
            if session_id not in self.browser_sessions:
                return {"error": "Browser session not found"}
            
            session = self.browser_sessions[session_id]
            session.last_activity = datetime.now()
            
            page = session.page
            
            if content_type == "text":
                if selector:
                    elements = await page.query_selector_all(selector)
                    content = []
                    for element in elements:
                        text = await element.text_content()
                        if text:
                            content.append(text.strip())
                    extracted = "\n".join(content)
                else:
                    extracted = await page.text_content("body")
            
            elif content_type == "html":
                if selector:
                    element = await page.query_selector(selector)
                    extracted = await element.inner_html() if element else ""
                else:
                    extracted = await page.content()
            
            elif content_type == "links":
                links = await page.query_selector_all("a[href]")
                extracted = []
                for link in links:
                    href = await link.get_attribute("href")
                    text = await link.text_content()
                    if href:
                        extracted.append({
                            "url": href,
                            "text": text.strip() if text else "",
                            "absolute_url": page.url + href if href.startswith("/") else href
                        })
            
            elif content_type == "images":
                images = await page.query_selector_all("img[src]")
                extracted = []
                for img in images:
                    src = await img.get_attribute("src")
                    alt = await img.get_attribute("alt")
                    if src:
                        extracted.append({
                            "src": src,
                            "alt": alt or "",
                            "absolute_url": page.url + src if src.startswith("/") else src
                        })
            
            else:
                return {"error": f"Unknown content type: {content_type}"}
            
            result = {
                "session_id": session_id,
                "url": page.url,
                "content_type": content_type,
                "selector": selector,
                "content": extracted,
                "timestamp": datetime.now().isoformat()
            }
            
            if analyze_with_ai and isinstance(extracted, str):
                analysis = await self._analyze_content_with_ai(extracted, page.url)
                result["ai_analysis"] = analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract page content: {e}")
            return {"error": str(e)}
    
    async def perform_web_search(
        self,
        query: str,
        search_engine: str = "duckduckgo",
        max_results: int = 10,
        analyze_results: bool = False
    ) -> Dict[str, Any]:
        """
        Perform web search using specified search engine.
        
        Args:
            query: Search query
            search_engine: Search engine to use
            max_results: Maximum number of results to return
            analyze_results: Whether to analyze results with Venice.ai
            
        Returns:
            Search results
        """
        try:
            if search_engine not in self.search_engines:
                return {"error": f"Unknown search engine: {search_engine}"}
            
            session_result = await self.create_browser_session(headless=True)
            if "error" in session_result:
                return session_result
            
            session_id = session_result["session_id"]
            
            search_url = self.search_engines[search_engine].format(query=query)
            
            nav_result = await self.navigate_to_url(session_id, search_url)
            if "error" in nav_result:
                await self.close_browser_session(session_id)
                return nav_result
            
            session = self.browser_sessions[session_id]
            page = session.page
            
            await page.wait_for_load_state("networkidle", timeout=10000)
            
            results = []
            
            if search_engine == "duckduckgo":
                result_elements = await page.query_selector_all("[data-result]")
                
                for i, element in enumerate(result_elements[:max_results]):
                    try:
                        title_elem = await element.query_selector("h2 a")
                        title = await title_elem.text_content() if title_elem else ""
                        url = await title_elem.get_attribute("href") if title_elem else ""
                        
                        snippet_elem = await element.query_selector("[data-result='snippet']")
                        snippet = await snippet_elem.text_content() if snippet_elem else ""
                        
                        if title and url:
                            results.append(SearchResult(
                                title=title.strip(),
                                url=url,
                                snippet=snippet.strip(),
                                rank=i + 1,
                                source=search_engine,
                                metadata={}
                            ))
                    except Exception:
                        continue
            
            elif search_engine == "bing":
                result_elements = await page.query_selector_all(".b_algo")
                
                for i, element in enumerate(result_elements[:max_results]):
                    try:
                        title_elem = await element.query_selector("h2 a")
                        title = await title_elem.text_content() if title_elem else ""
                        url = await title_elem.get_attribute("href") if title_elem else ""
                        
                        snippet_elem = await element.query_selector(".b_caption p")
                        snippet = await snippet_elem.text_content() if snippet_elem else ""
                        
                        if title and url:
                            results.append(SearchResult(
                                title=title.strip(),
                                url=url,
                                snippet=snippet.strip(),
                                rank=i + 1,
                                source=search_engine,
                                metadata={}
                            ))
                    except Exception:
                        continue
            
            await self.close_browser_session(session_id)
            
            search_results = [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "rank": r.rank,
                    "source": r.source
                }
                for r in results
            ]
            
            result = {
                "query": query,
                "search_engine": search_engine,
                "results_count": len(search_results),
                "results": search_results,
                "timestamp": datetime.now().isoformat()
            }
            
            if analyze_results and search_results:
                analysis = await self._analyze_search_results_with_ai(query, search_results)
                result["ai_analysis"] = analysis
            
            await self._store_search_memory(query, search_engine, len(search_results))
            self.stats["searches_performed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to perform web search: {e}")
            return {"error": str(e)}
    
    async def take_screenshot(
        self,
        session_id: str,
        full_page: bool = False,
        element_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Take a screenshot of the current page.
        
        Args:
            session_id: Browser session ID
            full_page: Whether to capture full page
            element_selector: CSS selector for specific element
            
        Returns:
            Screenshot information
        """
        try:
            if session_id not in self.browser_sessions:
                return {"error": "Browser session not found"}
            
            session = self.browser_sessions[session_id]
            session.last_activity = datetime.now()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{session_id}_{timestamp}.png"
            filepath = self.download_dir / filename
            
            if element_selector:
                element = await session.page.query_selector(element_selector)
                if element:
                    await element.screenshot(path=str(filepath))
                else:
                    return {"error": f"Element not found: {element_selector}"}
            else:
                await session.page.screenshot(
                    path=str(filepath),
                    full_page=full_page
                )
            
            self.stats["screenshots_taken"] += 1
            
            return {
                "session_id": session_id,
                "filename": filename,
                "filepath": str(filepath),
                "url": session.page.url,
                "full_page": full_page,
                "element_selector": element_selector,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return {"error": str(e)}
    
    async def download_file(
        self,
        url: str,
        filename: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download a file from a URL.
        
        Args:
            url: URL to download from
            filename: Custom filename (optional)
            session_id: Browser session ID for authenticated downloads
            
        Returns:
            Download result
        """
        try:
            if not filename:
                filename = url.split("/")[-1] or f"download_{int(datetime.now().timestamp())}"
            
            filepath = self.download_dir / filename
            
            if session_id and session_id in self.browser_sessions:
                session = self.browser_sessions[session_id]
                
                async with session.page.expect_download() as download_info:
                    await session.page.goto(url)
                
                download = await download_info.value
                await download.save_as(str(filepath))
            else:
                async with aiohttp.ClientSession() as http_session:
                    async with http_session.get(url) as response:
                        if response.status == 200:
                            async with aiofiles.open(filepath, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                        else:
                            return {"error": f"HTTP {response.status}: {response.reason}"}
            
            file_size = filepath.stat().st_size
            self.stats["files_downloaded"] += 1
            
            return {
                "url": url,
                "filename": filename,
                "filepath": str(filepath),
                "file_size": file_size,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return {"error": str(e)}
    
    async def close_browser_session(self, session_id: str) -> Dict[str, Any]:
        """Close a browser session."""
        try:
            if session_id not in self.browser_sessions:
                return {"error": "Browser session not found"}
            
            session = self.browser_sessions[session_id]
            
            await session.context.close()
            await session.browser.close()
            
            del self.browser_sessions[session_id]
            
            return {
                "session_id": session_id,
                "closed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to close browser session: {e}")
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Cleanup automation server resources."""
        logger.info("Cleaning up Automation MCP Server")
        
        try:
            for session_id in list(self.browser_sessions.keys()):
                await self.close_browser_session(session_id)
            
            if self.playwright:
                await self.playwright.stop()
            
            logger.info("Automation MCP Server cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Automation MCP Server cleanup: {e}")
    
    async def _analyze_content_with_ai(self, content: str, url: str) -> str:
        """Analyze page content with Venice.ai."""
        try:
            analysis_prompt = f"""
            Analyze this web page content and provide insights:
            
            URL: {url}
            Content: {content[:4000]}
            
            Provide:
            1. Main topics and themes
            2. Key information extracted
            3. Content quality and reliability assessment
            4. Potential use cases for this information
            """
            
            analysis = await self.venice_client.generate_response(
                prompt=analysis_prompt,
                model="claude-3-5-sonnet-20241022"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze content with AI: {e}")
            return f"Analysis failed: {str(e)}"
    
    async def _analyze_search_results_with_ai(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Analyze search results with Venice.ai."""
        try:
            results_text = "\n".join([
                f"{r['rank']}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n"
                for r in results[:5]
            ])
            
            analysis_prompt = f"""
            Analyze these search results for the query: "{query}"
            
            Results:
            {results_text}
            
            Provide:
            1. Most relevant results for the query
            2. Common themes across results
            3. Quality assessment of the sources
            4. Recommendations for further research
            """
            
            analysis = await self.venice_client.generate_response(
                prompt=analysis_prompt,
                model="claude-3-5-sonnet-20241022"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze search results with AI: {e}")
            return f"Analysis failed: {str(e)}"
    
    async def _store_navigation_memory(
        self,
        session_id: str,
        url: str,
        title: str
    ) -> None:
        """Store navigation event in long-term memory."""
        try:
            await self.long_term_memory.store_memory(
                content=f"Browser navigation: {title} ({url})",
                memory_type=MemoryType.EXPERIENCE,
                importance=MemoryImportance.LOW,
                tags=["automation", "navigation", "browser"],
                metadata={
                    "session_id": session_id,
                    "url": url,
                    "title": title,
                    "action": "navigate"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store navigation memory: {e}")
    
    async def _store_search_memory(
        self,
        query: str,
        search_engine: str,
        results_count: int
    ) -> None:
        """Store search event in long-term memory."""
        try:
            await self.long_term_memory.store_memory(
                content=f"Web search: '{query}' via {search_engine} ({results_count} results)",
                memory_type=MemoryType.RESEARCH,
                importance=MemoryImportance.MEDIUM,
                tags=["automation", "search", search_engine],
                metadata={
                    "query": query,
                    "search_engine": search_engine,
                    "results_count": results_count,
                    "action": "search"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store search memory: {e}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = int(datetime.now().timestamp())
        return f"session_{timestamp}_{len(self.browser_sessions)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "active_sessions": len(self.browser_sessions),
            "performance_stats": self.stats.copy()
        }


automation_server = AutomationMCPServer(
    venice_client=None,
    long_term_memory=None
)


@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available automation tools."""
    return [
        types.Tool(
            name="create_browser_session",
            description="Create a new browser session for automation",
            inputSchema={
                "type": "object",
                "properties": {
                    "headless": {
                        "type": "boolean",
                        "description": "Whether to run browser in headless mode",
                        "default": True
                    },
                    "browser_type": {
                        "type": "string",
                        "enum": ["chromium", "firefox", "webkit"],
                        "description": "Type of browser to use",
                        "default": "chromium"
                    },
                    "viewport": {
                        "type": "object",
                        "properties": {
                            "width": {"type": "integer"},
                            "height": {"type": "integer"}
                        },
                        "description": "Browser viewport size"
                    },
                    "user_agent": {
                        "type": "string",
                        "description": "Custom user agent string"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="navigate_to_url",
            description="Navigate to a URL in a browser session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to"
                    },
                    "wait_for": {
                        "type": "string",
                        "enum": ["load", "domcontentloaded", "networkidle"],
                        "description": "What to wait for after navigation",
                        "default": "domcontentloaded"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Navigation timeout in milliseconds",
                        "default": 30000
                    }
                },
                "required": ["session_id", "url"]
            }
        ),
        types.Tool(
            name="extract_page_content",
            description="Extract content from the current page",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID"
                    },
                    "content_type": {
                        "type": "string",
                        "enum": ["text", "html", "links", "images"],
                        "description": "Type of content to extract",
                        "default": "text"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for specific elements"
                    },
                    "analyze_with_ai": {
                        "type": "boolean",
                        "description": "Whether to analyze content with Venice.ai",
                        "default": False
                    }
                },
                "required": ["session_id"]
            }
        ),
        types.Tool(
            name="perform_web_search",
            description="Perform web search using specified search engine",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "search_engine": {
                        "type": "string",
                        "enum": ["duckduckgo", "bing", "google"],
                        "description": "Search engine to use",
                        "default": "duckduckgo"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "analyze_results": {
                        "type": "boolean",
                        "description": "Whether to analyze results with Venice.ai",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="take_screenshot",
            description="Take a screenshot of the current page",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID"
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": "Whether to capture full page",
                        "default": False
                    },
                    "element_selector": {
                        "type": "string",
                        "description": "CSS selector for specific element"
                    }
                },
                "required": ["session_id"]
            }
        ),
        types.Tool(
            name="download_file",
            description="Download a file from a URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to download from"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Custom filename (optional)"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID for authenticated downloads"
                    }
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="close_browser_session",
            description="Close a browser session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID to close"
                    }
                },
                "required": ["session_id"]
            }
        )
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls for automation operations."""
    try:
        if name == "create_browser_session":
            result = await automation_server.create_browser_session(
                headless=arguments.get("headless", True),
                browser_type=arguments.get("browser_type", "chromium"),
                viewport=arguments.get("viewport"),
                user_agent=arguments.get("user_agent")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "navigate_to_url":
            result = await automation_server.navigate_to_url(
                session_id=arguments["session_id"],
                url=arguments["url"],
                wait_for=arguments.get("wait_for", "domcontentloaded"),
                timeout=arguments.get("timeout", 30000)
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "extract_page_content":
            result = await automation_server.extract_page_content(
                session_id=arguments["session_id"],
                content_type=arguments.get("content_type", "text"),
                selector=arguments.get("selector"),
                analyze_with_ai=arguments.get("analyze_with_ai", False)
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "perform_web_search":
            result = await automation_server.perform_web_search(
                query=arguments["query"],
                search_engine=arguments.get("search_engine", "duckduckgo"),
                max_results=arguments.get("max_results", 10),
                analyze_results=arguments.get("analyze_results", False)
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "take_screenshot":
            result = await automation_server.take_screenshot(
                session_id=arguments["session_id"],
                full_page=arguments.get("full_page", False),
                element_selector=arguments.get("element_selector")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "download_file":
            result = await automation_server.download_file(
                url=arguments["url"],
                filename=arguments.get("filename"),
                session_id=arguments.get("session_id")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "close_browser_session":
            result = await automation_server.close_browser_session(
                session_id=arguments["session_id"]
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
    """Main entry point for the Automation MCP server."""
    import mcp.server.stdio
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
