"""
Template Engine for generating MCP server code.

This module provides templating capabilities for creating MCP servers
with different patterns and integrations based on problem requirements.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import jinja2
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ServerTemplate:
    """Template for generating MCP servers."""
    name: str
    description: str
    template_code: str
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemplateEngine:
    """
    Engine for generating MCP server code from templates.
    
    Provides flexible code generation capabilities for the Venice.ai
    scaffolding system to create specialized MCP servers.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template engine.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        self.jinja_env.filters['snake_case'] = self._snake_case
        self.jinja_env.filters['camel_case'] = self._camel_case
        self.jinja_env.filters['json_schema'] = self._json_schema_filter
        
        self.templates: Dict[str, ServerTemplate] = {}
        
        asyncio.create_task(self._initialize_builtin_templates())
    
    async def generate_server(
        self,
        name: str,
        tools: List[Any],  # ToolDesign objects
        integrations: Optional[List[str]] = None
    ) -> str:
        """
        Generate MCP server code for specialized tools.
        
        Args:
            name: Server name
            tools: List of tool designs
            integrations: List of integration points
            
        Returns:
            Generated server code
        """
        logger.info(f"Generating server code for {name} with {len(tools)} tools")
        
        try:
            template_name = self._select_template_for_tools(tools)
            template = await self.get_template(template_name)
            
            if not template:
                template_name = "basic_server"
                template = await self.get_template(template_name)
            
            context = {
                "server_name": name,
                "server_class": self._camel_case(name.replace("-", "_")),
                "tools": tools,
                "integrations": integrations or [],
                "timestamp": datetime.now().isoformat(),
                "dependencies": self._collect_dependencies(tools),
                "imports": self._generate_imports(tools, integrations),
                "tool_handlers": self._generate_tool_handlers(tools)
            }
            
            jinja_template = self.jinja_env.from_string(template.template_code)
            generated_code = jinja_template.render(**context)
            
            logger.debug(f"Generated {len(generated_code)} characters of server code")
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate server code for {name}: {e}")
            return self._generate_fallback_server(name, tools)
    
    async def generate_api_server(
        self,
        name: str,
        api_spec: Dict[str, Any],
        tools: List[Any]
    ) -> str:
        """
        Generate MCP server code for API integration.
        
        Args:
            name: Server name
            api_spec: OpenAPI specification
            tools: List of tool designs extracted from API
            
        Returns:
            Generated server code
        """
        logger.info(f"Generating API server code for {name}")
        
        try:
            template = await self.get_template("api_server")
            if not template:
                template = await self.get_template("basic_server")
            
            context = {
                "server_name": name,
                "server_class": self._camel_case(name.replace("-", "_")),
                "api_spec": api_spec,
                "api_base_url": api_spec.get("servers", [{}])[0].get("url", ""),
                "api_title": api_spec.get("info", {}).get("title", "API"),
                "api_version": api_spec.get("info", {}).get("version", "1.0.0"),
                "tools": tools,
                "timestamp": datetime.now().isoformat(),
                "auth_required": "security" in api_spec,
                "endpoints": self._extract_endpoints(api_spec)
            }
            
            jinja_template = self.jinja_env.from_string(template.template_code)
            generated_code = jinja_template.render(**context)
            
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate API server code for {name}: {e}")
            return self._generate_fallback_server(name, tools)
    
    async def generate_from_template(
        self,
        template_name: str,
        parameters: Dict[str, Any],
        server_name: str
    ) -> str:
        """
        Generate server code from a specific template.
        
        Args:
            template_name: Name of the template to use
            parameters: Template parameters
            server_name: Name of the server
            
        Returns:
            Generated server code
        """
        logger.info(f"Generating server from template {template_name}")
        
        try:
            template = await self.get_template(template_name)
            if not template:
                raise ValueError(f"Template {template_name} not found")
            
            context = {
                "server_name": server_name,
                "server_class": self._camel_case(server_name.replace("-", "_")),
                "timestamp": datetime.now().isoformat(),
                **template.parameters,
                **parameters
            }
            
            jinja_template = self.jinja_env.from_string(template.template_code)
            generated_code = jinja_template.render(**context)
            
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate from template {template_name}: {e}")
            raise
    
    async def add_template(self, template: ServerTemplate) -> bool:
        """
        Add a new template to the engine.
        
        Args:
            template: Template to add
            
        Returns:
            True if template added successfully
        """
        try:
            self.templates[template.name] = template
            
            template_file = self.templates_dir / f"{template.name}.py.j2"
            with open(template_file, 'w') as f:
                f.write(template.template_code)
            
            metadata_file = self.templates_dir / f"{template.name}.yaml"
            metadata = {
                "name": template.name,
                "description": template.description,
                "dependencies": template.dependencies,
                "parameters": template.parameters,
                "tags": template.tags,
                "metadata": template.metadata
            }
            
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            logger.info(f"Added template: {template.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add template {template.name}: {e}")
            return False
    
    async def get_template(self, template_name: str) -> Optional[ServerTemplate]:
        """
        Get a template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template or None if not found
        """
        if template_name in self.templates:
            return self.templates[template_name]
        
        try:
            template_file = self.templates_dir / f"{template_name}.py.j2"
            metadata_file = self.templates_dir / f"{template_name}.yaml"
            
            if template_file.exists() and metadata_file.exists():
                with open(template_file, 'r') as f:
                    template_code = f.read()
                
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                template = ServerTemplate(
                    name=metadata["name"],
                    description=metadata["description"],
                    template_code=template_code,
                    dependencies=metadata.get("dependencies", []),
                    parameters=metadata.get("parameters", {}),
                    tags=metadata.get("tags", []),
                    metadata=metadata.get("metadata", {})
                )
                
                self.templates[template_name] = template
                return template
                
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
        
        return None
    
    async def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of template information
        """
        templates = []
        
        for template in self.templates.values():
            templates.append({
                "name": template.name,
                "description": template.description,
                "dependencies": template.dependencies,
                "parameters": list(template.parameters.keys()),
                "tags": template.tags
            })
        
        return templates
    
    async def _initialize_builtin_templates(self):
        """Initialize built-in templates."""
        try:
            basic_template = ServerTemplate(
                name="basic_server",
                description="Basic MCP server with tool support",
                template_code=self._get_basic_server_template(),
                dependencies=["mcp"],
                parameters={
                    "description": "A basic MCP server",
                    "version": "1.0.0"
                },
                tags=["basic", "general"]
            )
            
            api_template = ServerTemplate(
                name="api_server",
                description="MCP server for API integration",
                template_code=self._get_api_server_template(),
                dependencies=["mcp", "aiohttp", "pydantic"],
                parameters={
                    "api_base_url": "",
                    "auth_required": False
                },
                tags=["api", "integration"]
            )
            
            scraping_template = ServerTemplate(
                name="web_scraping_server",
                description="MCP server for web scraping tasks",
                template_code=self._get_web_scraping_template(),
                dependencies=["mcp", "aiohttp", "beautifulsoup4", "selenium"],
                parameters={
                    "user_agent": "Mozilla/5.0 (compatible; MCP-WebScraper/1.0)",
                    "timeout": 30
                },
                tags=["web", "scraping", "automation"]
            )
            
            database_template = ServerTemplate(
                name="database_server",
                description="MCP server for database operations",
                template_code=self._get_database_template(),
                dependencies=["mcp", "sqlalchemy", "asyncpg"],
                parameters={
                    "database_url": "",
                    "pool_size": 10
                },
                tags=["database", "sql", "data"]
            )
            
            for template in [basic_template, api_template, scraping_template, database_template]:
                await self.add_template(template)
            
            logger.info("Initialized built-in templates")
            
        except Exception as e:
            logger.error(f"Failed to initialize built-in templates: {e}")
    
    def _select_template_for_tools(self, tools: List[Any]) -> str:
        """
        Select the most appropriate template based on tool characteristics.
        
        Args:
            tools: List of tool designs
            
        Returns:
            Template name
        """
        if not tools:
            return "basic_server"
        
        has_api_tools = any(
            getattr(tool, 'implementation_strategy', '') == 'api_call' 
            for tool in tools
        )
        
        has_web_tools = any(
            'web' in getattr(tool, 'tags', []) or 'scraping' in getattr(tool, 'tags', [])
            for tool in tools
        )
        
        has_database_tools = any(
            'database' in getattr(tool, 'tags', []) or 'sql' in getattr(tool, 'tags', [])
            for tool in tools
        )
        
        if has_api_tools:
            return "api_server"
        elif has_web_tools:
            return "web_scraping_server"
        elif has_database_tools:
            return "database_server"
        else:
            return "basic_server"
    
    def _collect_dependencies(self, tools: List[Any]) -> List[str]:
        """Collect all dependencies from tools."""
        dependencies = set(["mcp"])  # Always include MCP
        
        for tool in tools:
            tool_deps = getattr(tool, 'dependencies', [])
            dependencies.update(tool_deps)
        
        return sorted(list(dependencies))
    
    def _generate_imports(self, tools: List[Any], integrations: Optional[List[str]]) -> List[str]:
        """Generate import statements based on tools and integrations."""
        imports = [
            "import asyncio",
            "import logging",
            "from typing import Any, Dict, List",
            "from mcp.server.lowlevel import Server",
            "from mcp.server.models import InitializationOptions",
            "import mcp.types as types"
        ]
        
        for tool in tools:
            strategy = getattr(tool, 'implementation_strategy', '')
            if strategy == 'api_call':
                imports.extend([
                    "import aiohttp",
                    "import json"
                ])
            elif 'web' in getattr(tool, 'tags', []):
                imports.extend([
                    "import aiohttp",
                    "from bs4 import BeautifulSoup"
                ])
        
        if integrations:
            if 'venice_ai' in integrations:
                imports.append("from src.venice.client import VeniceClient")
            if 'memory' in integrations:
                imports.append("from src.memory.client import MemoryClient")
        
        return sorted(list(set(imports)))
    
    def _generate_tool_handlers(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Generate tool handler information."""
        handlers = []
        
        for tool in tools:
            handler = {
                "name": getattr(tool, 'name', 'unknown'),
                "description": getattr(tool, 'description', ''),
                "input_schema": getattr(tool, 'input_schema', {}),
                "implementation": self._generate_tool_implementation(tool)
            }
            handlers.append(handler)
        
        return handlers
    
    def _generate_tool_implementation(self, tool: Any) -> str:
        """Generate implementation code for a tool."""
        strategy = getattr(tool, 'implementation_strategy', '')
        
        if strategy == 'api_call':
            return self._generate_api_call_implementation(tool)
        elif 'web' in getattr(tool, 'tags', []):
            return self._generate_web_scraping_implementation(tool)
        else:
            return self._generate_basic_implementation(tool)
    
    def _generate_api_call_implementation(self, tool: Any) -> str:
        """Generate API call implementation."""
        metadata = getattr(tool, 'metadata', {})
        api_path = metadata.get('api_path', '/unknown')
        api_method = metadata.get('api_method', 'GET')
        
        return f'''
        async with aiohttp.ClientSession() as session:
            url = f"{{base_url}}{api_path}"
            async with session.{api_method.lower()}(url, json=arguments) as response:
                result = await response.json()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        '''
    
    def _generate_web_scraping_implementation(self, tool: Any) -> str:
        """Generate web scraping implementation."""
        return '''
        async with aiohttp.ClientSession() as session:
            url = arguments.get("url", "")
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                result = soup.get_text()
                return [types.TextContent(type="text", text=result)]
        '''
    
    def _generate_basic_implementation(self, tool: Any) -> str:
        """Generate basic implementation."""
        return '''
        result = f"Executed {name} with arguments: {arguments}"
        return [types.TextContent(type="text", text=result)]
        '''
    
    def _extract_endpoints(self, api_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract endpoint information from API spec."""
        endpoints = []
        paths = api_spec.get("paths", {})
        
        for path, methods in paths.items():
            for method, spec in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    endpoints.append({
                        "path": path,
                        "method": method.upper(),
                        "summary": spec.get("summary", ""),
                        "description": spec.get("description", ""),
                        "parameters": spec.get("parameters", [])
                    })
        
        return endpoints
    
    def _generate_fallback_server(self, name: str, tools: List[Any]) -> str:
        """Generate a basic fallback server."""
        return f'''
"""
Generated MCP server: {name}
Created at: {datetime.now().isoformat()}
"""

import asyncio
import logging
from typing import Any, Dict, List
from mcp.server.lowlevel import Server
import mcp.types as types

logger = logging.getLogger(__name__)

app = Server("{name}")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="echo",
            description="Echo the input arguments",
            inputSchema={{
                "type": "object",
                "properties": {{
                    "message": {{"type": "string", "description": "Message to echo"}}
                }},
                "required": ["message"]
            }}
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""
    if name == "echo":
        message = arguments.get("message", "No message provided")
        return [types.TextContent(type="text", text=f"Echo: {{message}}")]
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {{name}}")]

async def main():
    """Main server function."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="{name}",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={{}}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _get_basic_server_template(self) -> str:
        """Get the basic server template."""
        return '''
"""
Generated MCP server: {{ server_name }}
Created at: {{ timestamp }}
Description: {{ description }}
"""

{% for import in imports %}
{{ import }}
{% endfor %}

logger = logging.getLogger(__name__)

app = Server("{{ server_name }}")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        {% for handler in tool_handlers %}
        types.Tool(
            name="{{ handler.name }}",
            description="{{ handler.description }}",
            inputSchema={{ handler.input_schema | tojson }}
        ){% if not loop.last %},{% endif %}
        {% endfor %}
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""
    {% for handler in tool_handlers %}
    {% if loop.first %}if{% else %}elif{% endif %} name == "{{ handler.name }}":
        {{ handler.implementation | indent(8) }}
    {% endfor %}
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main server function."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="{{ server_name }}",
                server_version="{{ version }}",
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _get_api_server_template(self) -> str:
        """Get the API server template."""
        return '''
"""
Generated API MCP server: {{ server_name }}
API: {{ api_title }} v{{ api_version }}
Created at: {{ timestamp }}
"""

{% for import in imports %}
{{ import }}
{% endfor %}
import os

logger = logging.getLogger(__name__)

API_BASE_URL = "{{ api_base_url }}"
{% if auth_required %}
API_KEY = os.getenv("API_KEY", "")
{% endif %}

app = Server("{{ server_name }}")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available API tools."""
    return [
        {% for endpoint in endpoints %}
        types.Tool(
            name="{{ endpoint.method.lower() }}_{{ endpoint.path.replace('/', '_').replace('{', '').replace('}', '').strip('_') }}",
            description="{{ endpoint.summary or endpoint.description }}",
            inputSchema={
                "type": "object",
                "properties": {
                    {% for param in endpoint.parameters %}
                    "{{ param.name }}": {
                        "type": "{{ param.schema.type if param.schema else 'string' }}",
                        "description": "{{ param.description or '' }}"
                    }{% if not loop.last %},{% endif %}
                    {% endfor %}
                },
                "required": [{% for param in endpoint.parameters %}{% if param.required %}"{{ param.name }}"{% if not loop.last %}, {% endif %}{% endif %}{% endfor %}]
            }
        ){% if not loop.last %},{% endif %}
        {% endfor %}
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle API tool calls."""
    {% for endpoint in endpoints %}
    {% set tool_name = endpoint.method.lower() + '_' + endpoint.path.replace('/', '_').replace('{', '').replace('}', '').strip('_') %}
    {% if loop.first %}if{% else %}elif{% endif %} name == "{{ tool_name }}":
        async with aiohttp.ClientSession() as session:
            url = f"{API_BASE_URL}{{ endpoint.path }}"
            {% if auth_required %}
            headers = {"Authorization": f"Bearer {API_KEY}"}
            {% else %}
            headers = {}
            {% endif %}
            
            {% if endpoint.method.upper() == 'GET' %}
            async with session.get(url, headers=headers, params=arguments) as response:
            {% else %}
            async with session.{{ endpoint.method.lower() }}(url, headers=headers, json=arguments) as response:
            {% endif %}
                result = await response.json()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    {% endfor %}
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main server function."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="{{ server_name }}",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _get_web_scraping_template(self) -> str:
        """Get the web scraping template."""
        return '''
"""
Generated Web Scraping MCP server: {{ server_name }}
Created at: {{ timestamp }}
"""

{% for import in imports %}
{{ import }}
{% endfor %}
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

USER_AGENT = "{{ user_agent }}"
TIMEOUT = {{ timeout }}

app = Server("{{ server_name }}")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available web scraping tools."""
    return [
        types.Tool(
            name="scrape_page",
            description="Scrape content from a web page",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "selector": {"type": "string", "description": "CSS selector for content (optional)"}
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="extract_links",
            description="Extract all links from a web page",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract links from"}
                },
                "required": ["url"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle web scraping tool calls."""
    if name == "scrape_page":
        url = arguments.get("url", "")
        selector = arguments.get("selector", "")
        
        async with aiohttp.ClientSession() as session:
            headers = {"User-Agent": USER_AGENT}
            async with session.get(url, headers=headers, timeout=TIMEOUT) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                if selector:
                    elements = soup.select(selector)
                    result = "\\n".join([elem.get_text().strip() for elem in elements])
                else:
                    result = soup.get_text()
                
                return [types.TextContent(type="text", text=result)]
    
    elif name == "extract_links":
        url = arguments.get("url", "")
        
        async with aiohttp.ClientSession() as session:
            headers = {"User-Agent": USER_AGENT}
            async with session.get(url, headers=headers, timeout=TIMEOUT) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    text = link.get_text().strip()
                    links.append(f"{text}: {href}")
                
                result = "\\n".join(links)
                return [types.TextContent(type="text", text=result)]
    
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main server function."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="{{ server_name }}",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _get_database_template(self) -> str:
        """Get the database template."""
        return '''
"""
Generated Database MCP server: {{ server_name }}
Created at: {{ timestamp }}
"""

{% for import in imports %}
{{ import }}
{% endfor %}
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os

logger = logging.getLogger(__name__)

DATABASE_URL = "{{ database_url }}" or os.getenv("DATABASE_URL", "")
POOL_SIZE = {{ pool_size }}

app = Server("{{ server_name }}")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available database tools."""
    return [
        types.Tool(
            name="execute_query",
            description="Execute a SQL query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "params": {"type": "object", "description": "Query parameters (optional)"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="list_tables",
            description="List all tables in the database",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle database tool calls."""
    if not DATABASE_URL:
        return [types.TextContent(type="text", text="Database URL not configured")]
    
    engine = create_async_engine(DATABASE_URL, pool_size=POOL_SIZE)
    
    if name == "execute_query":
        query = arguments.get("query", "")
        params = arguments.get("params", {})
        
        async with engine.begin() as conn:
            result = await conn.execute(text(query), params)
            rows = result.fetchall()
            
            if rows:
                columns = result.keys()
                table_data = [list(columns)]
                for row in rows:
                    table_data.append(list(row))
                
                result_text = "\\n".join(["|".join(map(str, row)) for row in table_data])
            else:
                result_text = "Query executed successfully (no results)"
            
            return [types.TextContent(type="text", text=result_text)]
    
    elif name == "list_tables":
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result.fetchall()]
            
            result_text = "\\n".join(tables) if tables else "No tables found"
            return [types.TextContent(type="text", text=result_text)]
    
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main server function."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="{{ server_name }}",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _snake_case(self, text: str) -> str:
        """Convert text to snake_case."""
        return re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
    
    def _camel_case(self, text: str) -> str:
        """Convert text to CamelCase."""
        components = text.split('_')
        return ''.join(word.capitalize() for word in components)
    
    def _json_schema_filter(self, schema: Dict[str, Any]) -> str:
        """Convert schema to JSON string."""
        return json.dumps(schema, indent=2)
