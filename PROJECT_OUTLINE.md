# PDX Hackerspace AI Agent - Project Outline

## Overview
This project implements a Venice.ai scaffolding system for PDX Hackerspace operations - a comprehensive framework that empowers Venice.ai models to autonomously solve complex problems through intelligent research, dynamic tool creation, and coordinated execution. The system serves as an always-on oracle for hackerspace members, providing sophisticated assistance that goes beyond simple question-answering to actively problem-solving through adaptive tool generation.

### Core Concept: Venice.ai Scaffolding Architecture
The scaffolding provides a four-stage problem-solving framework that enables Venice.ai models to:

1. **Research Problems**: Systematically gather context through memory retrieval, web search, and knowledge synthesis to deeply understand challenges before attempting solutions
2. **Create Tools**: Dynamically generate specialized MCP (Model Context Protocol) servers and tools tailored to address specific identified needs, rather than relying on pre-built static tools
3. **Execute Solutions**: Orchestrate complex multi-step solutions through coordinated agent actions, tool calls, and task execution pipelines with intelligent retry and adaptation mechanisms
4. **Learn and Adapt**: Continuously store execution experiences, outcomes, and learned patterns in persistent memory to improve future problem-solving capabilities and tool creation decisions

### Scaffolding vs Traditional AI Systems
Unlike traditional AI systems that provide responses based on training data, this scaffolding enables Venice.ai models to:
- **Active Problem Solving**: Research unfamiliar domains and create custom solutions rather than providing generic responses
- **Tool Evolution**: Generate new capabilities on-demand rather than being limited to pre-programmed functions
- **Contextual Memory**: Build and maintain long-term understanding of the hackerspace environment, member needs, and successful solution patterns
- **Autonomous Operation**: Function independently as a continuous agent rather than requiring constant human prompting

## System Architecture

### Core Components

#### 1. Agent Core (`src/agent/`)
- **Agent Manager** (`manager.py`) - Central orchestrator for Venice.ai scaffolding operations, coordinating research, tool creation, and solution execution
- **Session Manager** (`session.py`) - Maintains unified context across interfaces and tool interactions, preserving problem-solving state
- **Task Executor** (`executor.py`) - Handles background task execution with intelligent retry logic and result storage in memory
- **Reflector** (`reflector.py`) - Analyzes tool usage patterns, solution effectiveness, and learns from execution outcomes
- **Tool Creator** (`tool_creator.py`) - **Core scaffolding component** that dynamically generates specialized MCP servers and tools based on problem analysis and research findings
- **Research Coordinator** (`research_coordinator.py`) - Orchestrates multi-source research including memory retrieval, web search, and knowledge synthesis

#### 2. Memory System (`src/memory/`)
- **Vector Database Integration** (`vector_store.py`) - Chroma DB for embeddings and semantic search
- **Long-term Memory** (`long_term_memory.py`) - Persistent memory retrieval and experience storage
- **Context Manager** (`context.py`) - Intelligent context pruning, retrieval, and relevance scoring
- **Memory Reflection** (`reflection.py`) - Daily activity summarization and pattern recognition
- **Research Memory** (`research_memory.py`) - **Scaffolding component** that stores and indexes research findings, problem-solution mappings, and tool creation decisions
- **Experience Database** (`experience_db.py`) - Tracks tool usage effectiveness, solution outcomes, and learning patterns for future problem-solving

#### 3. Venice.ai Integration (`src/venice/`)
- **Venice Client** (`client.py`) - API wrapper for Venice.ai LLM calls
- **Model Manager** (`models.py`) - Handle different model types (Llama 4, Qwen QwQ-32B)
- **Token Management** (`tokens.py`) - Context window and token optimization

#### 4. MCP Framework (`src/mcp/`)
- **MCP Server Manager** (`server_manager.py`) - Lifecycle management for MCP servers including persistent services and on-the-fly creation
- **Tool Registry** (`registry.py`) - Discovery and cataloging of available MCP tools, resources, and their capabilities
- **MCP Client** (`client.py`) - Client for communicating with MCP servers via JSON-RPC protocol
- **Transport Manager** (`transport.py`) - Handle stdio and SSE transport protocols for different deployment scenarios
- **Server Factory** (`factory.py`) - **Core scaffolding component** that dynamically creates and configures new MCP servers based on Venice.ai model requirements and problem analysis
- **Capability Analyzer** (`capability_analyzer.py`) - Analyzes existing tools and identifies gaps that require new MCP server creation
- **Template Engine** (`template_engine.py`) - Generates MCP server code templates for common tool patterns and integrations

#### 5. MCP Server Integration (`src/mcp_servers/`)
- **Slack MCP Server** (`slack_server.py`) - MCP server exposing Slack tools (send_message, get_channel_history, manage_channels)
- **Discord MCP Server** (`discord_server.py`) - MCP server exposing Discord tools (send_message, get_guild_info, manage_roles)
- **Infrastructure MCP Server** (`infrastructure_server.py`) - MCP server for system monitoring tools (check_server_status, get_metrics, restart_services)
- **Automation MCP Server** (`automation_server.py`) - MCP server for browser automation and web search tools (web_search, browser_navigate, file_operations)
- **Web Portal** (`web_portal.py`) - Web-based interface for direct user interaction
- **API Server** (`api_server.py`) - REST API for external integrations

#### 6. Infrastructure Monitoring (`src/monitoring/`)
- **System Monitor** (`system_monitor.py`) - Server and network monitoring
- **Prometheus Integration** (`prometheus.py`) - Metrics collection
- **Home Assistant** (`home_assistant.py`) - IoT device integration
- **Network Scanner** (`network.py`) - Network device monitoring

#### 7. Research Pipeline (`src/research/`)
- **Web Search Engine** (`web_search.py`) - **Core scaffolding component** for intelligent web search with result synthesis and relevance scoring
- **Knowledge Synthesizer** (`knowledge_synthesizer.py`) - Combines information from multiple sources (memory, web, documents) into coherent understanding
- **Domain Analyzer** (`domain_analyzer.py`) - Analyzes unfamiliar problem domains and identifies key concepts and relationships
- **Research Planner** (`research_planner.py`) - Creates systematic research strategies based on problem complexity and available information sources
- **Information Validator** (`information_validator.py`) - Validates and cross-references information from different sources for accuracy

#### 8. Automation Tools (`src/automation/`)
- **Browser Automation** (`browser.py`) - Puppeteer/Firecrawl integration for web interaction and data extraction
- **File System** (`filesystem.py`) - Safe file operations with sandboxing and permission controls
- **Shell Access** (`shell.py`) - Controlled shell command execution with security constraints
- **API Integration** (`api_integration.py`) - Generic API client framework for integrating with external services

## Hardware Considerations

### VRAM Constraints
- **Global VRAM Limit**: 64GB shared across all models
- **Model Selection**: Optimize for efficiency within VRAM constraints
- **Memory Management**: Implement model swapping and caching strategies
- **Monitoring**: Track VRAM usage and implement alerts

### Performance Optimization
- **Model Quantization**: Use quantized models when appropriate
- **Batch Processing**: Optimize inference batching
- **Caching**: Implement intelligent response caching
- **Load Balancing**: Distribute workload across available resources

## Security Framework

### Access Control
- **Authentication**: Secure API key management for Venice.ai
- **Authorization**: Role-based access for different interfaces
- **Sandboxing**: Isolated execution environments for tools
- **Audit Logging**: Comprehensive activity logging

### Data Protection
- **Encryption**: Encrypt sensitive data at rest and in transit
- **Privacy**: Implement data retention policies
- **Compliance**: Follow hackerspace privacy guidelines
- **Backup**: Regular backup of memory and configuration data

## Configuration Management

### Environment Configuration (`config/`)
- **Development** (`dev.yaml`) - Local development settings
- **Production** (`prod.yaml`) - Production deployment settings
- **Secrets** (`secrets.yaml.example`) - Template for sensitive data
- **MCP Servers** (`mcp_servers.yaml`) - MCP server configurations

### Feature Flags
- **Interface Toggles** - Enable/disable specific interfaces
- **Tool Permissions** - Granular tool access control
- **Memory Settings** - Configurable memory retention policies
- **Monitoring Levels** - Adjustable monitoring verbosity

## MCP Server Architecture

### MCP Server Implementation Patterns
Each MCP server in the Venice.ai scaffolding follows the standardized Model Context Protocol, enabling dynamic tool creation and execution:

#### Core MCP Server Structure
```python
from mcp.server.lowlevel import Server
from mcp.server.models import InitializationOptions
import mcp.types as types
import asyncio
from typing import Any, Dict, List

# Initialize MCP server with descriptive name
app = Server("venice-scaffolding-server")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Expose available tools to the Venice.ai scaffolding system."""
    return [
        types.Tool(
            name="research_problem",
            description="Research a problem using web search and memory retrieval",
            inputSchema={
                "type": "object",
                "properties": {
                    "problem_description": {"type": "string", "description": "Description of the problem to research"},
                    "search_depth": {"type": "string", "enum": ["shallow", "deep"], "description": "Depth of research to perform"}
                },
                "required": ["problem_description"]
            }
        ),
        types.Tool(
            name="create_specialized_tool",
            description="Dynamically create a new MCP server with specialized tools",
            inputSchema={
                "type": "object",
                "properties": {
                    "tool_purpose": {"type": "string", "description": "Purpose and functionality of the new tool"},
                    "tool_schema": {"type": "object", "description": "JSON schema defining the tool's input parameters"}
                },
                "required": ["tool_purpose", "tool_schema"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute tools with Venice.ai integration for intelligent processing."""
    if name == "research_problem":
        # Research pipeline: memory retrieval + web search + synthesis
        research_result = await conduct_research(
            problem=arguments["problem_description"],
            depth=arguments.get("search_depth", "shallow")
        )
        return [types.TextContent(type="text", text=research_result)]
    
    elif name == "create_specialized_tool":
        # Dynamic tool creation based on research findings
        new_server_config = await generate_mcp_server(
            purpose=arguments["tool_purpose"],
            schema=arguments["tool_schema"]
        )
        return [types.TextContent(type="text", text=f"Created specialized tool: {new_server_config}")]
    
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def conduct_research(problem: str, depth: str) -> str:
    """Implement research pipeline with Venice.ai integration."""
    # This would integrate with Venice.ai API for intelligent research
    return f"Research completed for: {problem} (depth: {depth})"

async def generate_mcp_server(purpose: str, schema: dict) -> str:
    """Generate new MCP server based on identified needs."""
    # This would dynamically create new MCP servers
    return f"Generated MCP server for: {purpose}"
```

#### Slack Integration MCP Server Example
```python
from mcp.server.lowlevel import Server
import mcp.types as types
from slack_sdk.web.async_client import AsyncWebClient

app = Server("slack-integration-server")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="send_message",
            description="Send message to Slack channel with Venice.ai processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Slack channel ID or name"},
                    "message": {"type": "string", "description": "Message content to send"},
                    "process_with_ai": {"type": "boolean", "description": "Whether to process message with Venice.ai first"}
                },
                "required": ["channel", "message"]
            }
        ),
        types.Tool(
            name="analyze_channel_sentiment",
            description="Analyze sentiment and topics in a Slack channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Slack channel to analyze"},
                    "time_range": {"type": "string", "description": "Time range for analysis (e.g., '24h', '7d')"}
                },
                "required": ["channel"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "send_message":
        # Integrate with Venice.ai for message processing if requested
        if arguments.get("process_with_ai", False):
            processed_message = await process_with_venice_ai(arguments["message"])
        else:
            processed_message = arguments["message"]
        
        # Send to Slack
        result = await send_slack_message(arguments["channel"], processed_message)
        return [types.TextContent(type="text", text=result)]
    
    elif name == "analyze_channel_sentiment":
        # Retrieve channel history and analyze with Venice.ai
        analysis = await analyze_channel_with_venice_ai(
            arguments["channel"], 
            arguments.get("time_range", "24h")
        )
        return [types.TextContent(type="text", text=analysis)]
```

### Transport Options and Deployment
- **stdio Transport** (default): `python -m mcp_servers.slack_server` for direct process communication
- **SSE Transport**: `python -m mcp_servers.slack_server --transport sse --port 8000` for HTTP-based communication
- **Docker Deployment**: Containerized MCP servers for scalable deployment
- **Process Management**: Systemd services for persistent server operation

### Venice.ai Integration Architecture
- **API Client**: Centralized Venice.ai client for LLM operations, embeddings, and memory storage
- **Memory Interface**: Tools access vector database and long-term memory for context retrieval
- **Agent Coordination**: MCP tools coordinate with agent system for complex task execution
- **Dynamic Tool Creation**: System spawns new MCP servers when Venice.ai models identify capability gaps
- **Research Pipeline**: Integrated web search, memory retrieval, and knowledge synthesis capabilities

### MCP Server Lifecycle Management
- **Persistent Services**: Long-running MCP servers for core functionality (Slack, Discord, Infrastructure monitoring)
- **On-the-fly Creation**: Dynamic server spawning for specialized tasks or temporary operations discovered during problem research
- **Server Registry**: Central discovery system tracking available MCP servers, their capabilities, and health status
- **Health Monitoring**: Automatic restart, health checks, and performance monitoring for MCP server processes
- **Resource Management**: Intelligent allocation of computational resources based on server usage patterns

## Dynamic Tool Creation Architecture

The Venice.ai scaffolding system's core innovation is its ability to dynamically create specialized tools and MCP servers based on problem analysis and research findings. This architecture enables the system to evolve and adapt to new challenges without requiring manual tool development.

### Tool Creation Pipeline

#### 1. Problem Analysis and Gap Identification
```python
class CapabilityAnalyzer:
    async def analyze_problem_requirements(self, problem_description: str) -> ToolRequirements:
        """Analyze a problem to identify required capabilities and tool gaps."""
        # Use Venice.ai to understand problem domain and requirements
        problem_analysis = await self.venice_client.analyze_problem(problem_description)
        
        # Check existing tool registry for capability coverage
        existing_capabilities = await self.tool_registry.get_available_capabilities()
        
        # Identify gaps between required and available capabilities
        capability_gaps = self.identify_gaps(problem_analysis.requirements, existing_capabilities)
        
        return ToolRequirements(
            domain=problem_analysis.domain,
            required_actions=problem_analysis.actions,
            missing_capabilities=capability_gaps,
            complexity_score=problem_analysis.complexity
        )
```

#### 2. Research-Driven Tool Design
```python
class ToolDesigner:
    async def design_tool_from_research(self, requirements: ToolRequirements) -> ToolDesign:
        """Design a new tool based on research findings and requirements."""
        # Conduct domain-specific research
        research_results = await self.research_coordinator.research_domain(
            domain=requirements.domain,
            focus_areas=requirements.required_actions
        )
        
        # Synthesize research into tool specifications
        tool_spec = await self.venice_client.synthesize_tool_design(
            requirements=requirements,
            research_context=research_results,
            existing_patterns=await self.get_successful_tool_patterns()
        )
        
        return ToolDesign(
            name=tool_spec.name,
            description=tool_spec.description,
            input_schema=tool_spec.schema,
            implementation_strategy=tool_spec.strategy,
            integration_points=tool_spec.integrations
        )
```

#### 3. Dynamic MCP Server Generation
```python
class ServerFactory:
    async def create_specialized_server(self, tool_design: ToolDesign) -> MCPServerConfig:
        """Generate a new MCP server with specialized tools."""
        # Generate server code from template
        server_code = await self.template_engine.generate_server(
            name=f"{tool_design.name}-server",
            tools=[tool_design],
            integrations=tool_design.integration_points
        )
        
        # Create server configuration
        server_config = MCPServerConfig(
            name=f"{tool_design.name}-server",
            code=server_code,
            dependencies=tool_design.dependencies,
            transport="stdio",  # Default to stdio transport
            lifecycle="on-demand"  # Can be "persistent" or "on-demand"
        )
        
        # Deploy and register the new server
        await self.deploy_server(server_config)
        await self.tool_registry.register_server(server_config)
        
        return server_config
```

### Integration with Research Pipeline

#### Research-First Approach
The system follows a research-first methodology where Venice.ai models:

1. **Domain Research**: Investigate unfamiliar problem domains through web search, documentation analysis, and knowledge synthesis
2. **Solution Pattern Analysis**: Study existing solutions and identify successful patterns and approaches
3. **Tool Requirement Specification**: Define precise tool requirements based on research findings
4. **Implementation Strategy**: Develop implementation approaches informed by research and best practices

#### Knowledge-Driven Tool Creation
```python
class ResearchIntegratedToolCreator:
    async def create_tool_from_problem(self, problem: str) -> CreatedTool:
        """Complete pipeline from problem to deployed tool."""
        
        # Step 1: Research the problem domain
        research_context = await self.research_pipeline.comprehensive_research(
            problem=problem,
            depth="deep",
            sources=["web", "memory", "documentation"]
        )
        
        # Step 2: Analyze capability requirements
        requirements = await self.capability_analyzer.analyze_problem_requirements(
            problem_description=problem,
            research_context=research_context
        )
        
        # Step 3: Design tool based on research
        tool_design = await self.tool_designer.design_tool_from_research(
            requirements=requirements,
            research_findings=research_context
        )
        
        # Step 4: Generate and deploy MCP server
        server_config = await self.server_factory.create_specialized_server(tool_design)
        
        # Step 5: Test and validate the new tool
        validation_result = await self.tool_validator.validate_tool(
            server_config=server_config,
            test_cases=requirements.test_scenarios
        )
        
        # Step 6: Store creation experience for learning
        await self.experience_database.store_tool_creation(
            problem=problem,
            research_context=research_context,
            tool_design=tool_design,
            validation_result=validation_result,
            success_metrics=validation_result.performance_metrics
        )
        
        return CreatedTool(
            server_config=server_config,
            capabilities=tool_design.capabilities,
            performance_metrics=validation_result.metrics
        )
```

### Lifecycle Management for Dynamic Tools

#### On-Demand vs Persistent Servers
- **On-Demand Servers**: Created for specific tasks, automatically cleaned up after completion
- **Persistent Servers**: Promoted from on-demand based on usage patterns and success rates
- **Server Evolution**: Existing servers can be enhanced with new tools based on usage analysis

#### Learning and Optimization
```python
class ToolEvolutionManager:
    async def optimize_tool_ecosystem(self):
        """Continuously improve the tool ecosystem based on usage patterns."""
        
        # Analyze tool usage and success rates
        usage_analytics = await self.analytics_engine.analyze_tool_performance()
        
        # Identify underperforming or redundant tools
        optimization_candidates = await self.identify_optimization_opportunities(usage_analytics)
        
        # Evolve successful tools and retire unsuccessful ones
        for candidate in optimization_candidates:
            if candidate.action == "enhance":
                await self.enhance_tool(candidate.tool, candidate.improvements)
            elif candidate.action == "retire":
                await self.retire_tool(candidate.tool)
            elif candidate.action == "merge":
                await self.merge_tools(candidate.tools, candidate.merged_design)
```

### Security and Sandboxing

#### Safe Tool Execution
- **Sandboxed Environments**: All dynamically created tools run in isolated environments
- **Permission Controls**: Strict permission models for file system, network, and system access
- **Code Review**: Automated analysis of generated tool code for security vulnerabilities
- **Resource Limits**: CPU, memory, and execution time limits for dynamic tools

#### Validation and Testing
- **Automated Testing**: Generated tools include comprehensive test suites
- **Performance Monitoring**: Continuous monitoring of tool performance and resource usage
- **Rollback Capabilities**: Ability to quickly disable or rollback problematic tools

## Development Workflow

### Project Structure
```
ai-core/
├── src/
│   ├── agent/          # Core agent logic
│   ├── memory/         # Memory management
│   ├── venice/         # Venice.ai integration
│   ├── mcp/           # MCP framework and registry
│   ├── mcp_servers/   # MCP server implementations
│   ├── interfaces/    # Web portal and API interfaces
│   ├── monitoring/    # Infrastructure monitoring
│   └── automation/    # Automation tools
├── config/            # Configuration files
├── tests/            # Test suites
├── docs/             # Documentation
├── scripts/          # Utility scripts
└── docker/           # Container configurations
```

### Testing Strategy
- **Unit Tests** - Individual component testing
- **Integration Tests** - Cross-component functionality
- **MCP Tool Tests** - Tool integration validation
- **Performance Tests** - VRAM and latency benchmarks
- **Security Tests** - Vulnerability assessments

### Deployment Strategy
- **Containerization** - Docker containers for consistent deployment
- **Service Management** - Systemd services for continuous operation
- **Health Checks** - Automated health monitoring
- **Rollback Procedures** - Safe deployment rollback mechanisms

## Implementation Phases

### Phase 1: Scaffolding Foundation (Weeks 1-2)
- Set up project structure and development environment for Venice.ai scaffolding system
- Implement Venice.ai client integration with agent coordination and dynamic problem-solving capabilities
- Create core agent manager with research coordinator, tool creator, and task executor architecture
- Basic memory system with vector database (Chroma DB) for research findings and experience storage
- MCP framework foundation (server manager, transport layer, dynamic server creation capabilities)

### Phase 2: Core Tool Scaffolding (Weeks 3-4)
- Implement Slack MCP Server with tools for message handling, channel analysis, and sentiment monitoring
- Implement Discord MCP Server with guild management, role administration, and community interaction tools
- Create MCP server registry and discovery system enabling dynamic tool creation and capability tracking
- Integrate MCP servers with Venice.ai client for intelligent, model-driven tool execution and coordination
- Implement security framework and access controls for dynamically created MCP tools with sandboxing
- Add research tools (web search engine, knowledge synthesizer, domain analyzer) for comprehensive problem analysis

### Phase 3: Advanced Scaffolding Capabilities (Weeks 5-6)
- Infrastructure MCP Server for system monitoring, health checks, and Prometheus integration
- Automation MCP Server for browser automation, web interaction, and data extraction capabilities
- Tool Creator system enabling Venice.ai models to generate specialized MCP servers on-demand based on problem analysis
- Create web portal interface for direct user interaction, tool management, and scaffolding system monitoring
- Add memory reflection and learning capabilities to continuously improve tool creation decisions and success rates over time

### Phase 4: Production Scaffolding (Weeks 7-8)
- MCP server orchestration and lifecycle management for dynamic tool creation with automated scaling and cleanup
- Performance optimization and VRAM management for model operations, tool execution, and concurrent scaffolding processes
- Comprehensive testing of complete scaffolding system: problem research → tool creation → solution execution → learning cycle
- Documentation and deployment procedures for the Venice.ai scaffolding ecosystem including tool creation guidelines
- Monitoring and alerting setup for scaffolding health, tool creation success rates, and overall system performance metrics

## Success Metrics

### Functional Metrics
- **Response Time** - Average response time < 2 seconds
- **Uptime** - 99.9% availability target
- **Memory Efficiency** - Effective context retention and retrieval
- **Tool Success Rate** - >95% successful tool executions

### User Experience Metrics
- **User Satisfaction** - Regular feedback collection
- **Feature Adoption** - Track interface usage patterns
- **Error Rates** - Monitor and minimize user-facing errors
- **Learning Effectiveness** - Measure improvement in responses over time

## Risk Mitigation

### Technical Risks
- **VRAM Limitations** - Implement model swapping and optimization
- **API Rate Limits** - Implement request queuing and retry logic
- **Network Failures** - Robust error handling and offline capabilities
- **Data Loss** - Regular backups and redundancy

### Operational Risks
- **Security Breaches** - Comprehensive security framework
- **Service Outages** - High availability architecture
- **Resource Exhaustion** - Monitoring and auto-scaling
- **Configuration Errors** - Validation and testing procedures

## Future Enhancements

### Advanced AI Capabilities
- **Multi-modal Processing** - Image and document analysis
- **Predictive Analytics** - Proactive system monitoring
- **Natural Language Interface** - Conversational system administration
- **Learning Optimization** - Continuous improvement algorithms

### Integration Expansions
- **Additional Platforms** - Matrix, IRC, email integration
- **IoT Expansion** - More Home Assistant integrations
- **Cloud Services** - AWS, GCP, Azure monitoring
- **Development Tools** - GitHub, GitLab, CI/CD integration

This outline provides a comprehensive roadmap for implementing the continuous AI agent system while considering hardware constraints, security requirements, and scalability needs.
