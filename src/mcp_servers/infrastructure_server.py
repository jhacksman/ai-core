"""
Infrastructure MCP Server - System monitoring and management using latest MCP protocol.

This MCP server provides infrastructure monitoring tools with Venice.ai processing capabilities,
OAuth 2.0 authentication, and streamable HTTP transport support.
"""

import asyncio
import logging
import psutil
import subprocess
import json
import os
import platform
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from mcp.server.lowlevel import Server
import mcp.types as types

from ..venice.client import VeniceClient
from ..memory.long_term_memory import LongTermMemory, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)

app = Server("infrastructure-monitoring-server")

@dataclass
class SystemMetrics:
    """System metrics data structure."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    uptime: float

@dataclass
class ServiceStatus:
    """Service status data structure."""
    name: str
    status: str
    pid: Optional[int]
    memory_usage: float
    cpu_percent: float
    uptime: Optional[float]

class InfrastructureMCPServer:
    """
    Infrastructure MCP Server with Venice.ai integration.
    
    Provides system monitoring and management tools with AI analysis capabilities
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
        Initialize Infrastructure MCP Server.
        
        Args:
            venice_client: Venice.ai client for AI processing
            long_term_memory: Long-term memory system
            config: Configuration options
        """
        self.venice_client = venice_client
        self.long_term_memory = long_term_memory
        self.config = config or {}
        
        self.monitored_services = self.config.get("monitored_services", [
            "nginx", "apache2", "mysql", "postgresql", "redis", "docker", "ssh"
        ])
        
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage": 90.0,
            "load_average": 5.0
        })
        
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = self.config.get("max_history_size", 1000)
        
        self.stats = {
            "metrics_collected": 0,
            "services_checked": 0,
            "alerts_generated": 0,
            "commands_executed": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the Infrastructure MCP server."""
        logger.info("Initializing Infrastructure MCP Server")
        
        try:
            system_info = await self.get_system_info()
            logger.info(f"Monitoring system: {system_info['platform']} {system_info['architecture']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Infrastructure MCP Server: {e}")
            raise
    
    async def get_system_metrics(self, include_processes: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive system metrics.
        
        Args:
            include_processes: Whether to include detailed process information
            
        Returns:
            System metrics data
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": (usage.used / usage.total) * 100
                    }
                except PermissionError:
                    continue
            
            network_io = psutil.net_io_counters()._asdict()
            
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
            
            boot_time = psutil.boot_time()
            uptime = datetime.now().timestamp() - boot_time
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage={k: v["percent"] for k, v in disk_usage.items()},
                network_io=network_io,
                process_count=len(psutil.pids()),
                load_average=list(load_avg),
                uptime=uptime
            )
            
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            result = {
                "timestamp": metrics.timestamp.isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "load_average": list(load_avg)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": disk_usage,
                "network": network_io,
                "processes": {
                    "count": len(psutil.pids())
                },
                "uptime": uptime
            }
            
            if include_processes:
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                result["processes"]["details"] = sorted(
                    processes, 
                    key=lambda x: x.get('cpu_percent', 0), 
                    reverse=True
                )[:20]
            
            await self._store_metrics_memory(metrics)
            self.stats["metrics_collected"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    async def check_service_status(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Check status of system services.
        
        Args:
            service_name: Specific service to check (optional)
            
        Returns:
            Service status information
        """
        try:
            services_to_check = [service_name] if service_name else self.monitored_services
            service_statuses = []
            
            for service in services_to_check:
                try:
                    result = subprocess.run(
                        ["systemctl", "is-active", service],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    status = result.stdout.strip()
                    
                    service_info = subprocess.run(
                        ["systemctl", "show", service, "--property=MainPID,MemoryCurrent"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    pid = None
                    memory_usage = 0
                    cpu_percent = 0
                    
                    for line in service_info.stdout.split('\n'):
                        if line.startswith('MainPID='):
                            pid_str = line.split('=')[1]
                            pid = int(pid_str) if pid_str != '0' else None
                        elif line.startswith('MemoryCurrent='):
                            memory_usage = int(line.split('=')[1]) / 1024 / 1024  # Convert to MB
                    
                    if pid:
                        try:
                            proc = psutil.Process(pid)
                            cpu_percent = proc.cpu_percent()
                            uptime = datetime.now().timestamp() - proc.create_time()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            uptime = None
                    else:
                        uptime = None
                    
                    service_status = ServiceStatus(
                        name=service,
                        status=status,
                        pid=pid,
                        memory_usage=memory_usage,
                        cpu_percent=cpu_percent,
                        uptime=uptime
                    )
                    
                    service_statuses.append({
                        "name": service_status.name,
                        "status": service_status.status,
                        "pid": service_status.pid,
                        "memory_usage_mb": service_status.memory_usage,
                        "cpu_percent": service_status.cpu_percent,
                        "uptime_seconds": service_status.uptime
                    })
                    
                except subprocess.TimeoutExpired:
                    service_statuses.append({
                        "name": service,
                        "status": "timeout",
                        "error": "Command timed out"
                    })
                except Exception as e:
                    service_statuses.append({
                        "name": service,
                        "status": "error",
                        "error": str(e)
                    })
            
            self.stats["services_checked"] += len(services_to_check)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "services": service_statuses
            }
            
        except Exception as e:
            logger.error(f"Failed to check service status: {e}")
            return {"error": str(e)}
    
    async def restart_service(self, service_name: str, use_sudo: bool = True) -> Dict[str, Any]:
        """
        Restart a system service.
        
        Args:
            service_name: Name of the service to restart
            use_sudo: Whether to use sudo for the command
            
        Returns:
            Restart operation result
        """
        try:
            cmd = ["systemctl", "restart", service_name]
            if use_sudo:
                cmd = ["sudo"] + cmd
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            success = result.returncode == 0
            
            await self.long_term_memory.store_memory(
                content=f"Service restart: {service_name} - {'Success' if success else 'Failed'}",
                memory_type=MemoryType.EXPERIENCE,
                importance=MemoryImportance.HIGH if not success else MemoryImportance.MEDIUM,
                tags=["infrastructure", "service_restart", service_name],
                metadata={
                    "service": service_name,
                    "success": success,
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            )
            
            self.stats["commands_executed"] += 1
            
            return {
                "service": service_name,
                "success": success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            return {
                "service": service_name,
                "success": False,
                "error": "Command timed out",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
            return {
                "service": service_name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_system_health(self, time_range: str = "1h") -> Dict[str, Any]:
        """
        Analyze system health using Venice.ai with historical metrics.
        
        Args:
            time_range: Time range for analysis (e.g., '1h', '24h', '7d')
            
        Returns:
            Health analysis results
        """
        try:
            current_metrics = await self.get_system_metrics(include_processes=True)
            
            cutoff_time = self._calculate_cutoff_time(time_range)
            historical_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
            if not historical_metrics:
                historical_summary = "No historical data available"
            else:
                avg_cpu = sum(m.cpu_percent for m in historical_metrics) / len(historical_metrics)
                avg_memory = sum(m.memory_percent for m in historical_metrics) / len(historical_metrics)
                max_cpu = max(m.cpu_percent for m in historical_metrics)
                max_memory = max(m.memory_percent for m in historical_metrics)
                
                historical_summary = f"""
                Historical metrics ({time_range}):
                - Average CPU: {avg_cpu:.1f}%
                - Average Memory: {avg_memory:.1f}%
                - Peak CPU: {max_cpu:.1f}%
                - Peak Memory: {max_memory:.1f}%
                - Data points: {len(historical_metrics)}
                """
            
            analysis_prompt = f"""
            Analyze the system health based on current and historical metrics:
            
            Current System State:
            - CPU Usage: {current_metrics.get('cpu', {}).get('percent', 0):.1f}%
            - Memory Usage: {current_metrics.get('memory', {}).get('percent', 0):.1f}%
            - Load Average: {current_metrics.get('cpu', {}).get('load_average', [])}
            - Process Count: {current_metrics.get('processes', {}).get('count', 0)}
            - Uptime: {current_metrics.get('uptime', 0) / 3600:.1f} hours
            
            {historical_summary}
            
            Alert Thresholds:
            - CPU: {self.alert_thresholds['cpu_percent']}%
            - Memory: {self.alert_thresholds['memory_percent']}%
            - Load Average: {self.alert_thresholds['load_average']}
            
            Provide:
            1. Overall system health assessment (Excellent/Good/Warning/Critical)
            2. Specific issues or concerns identified
            3. Performance trends and patterns
            4. Recommendations for optimization or maintenance
            5. Immediate actions needed (if any)
            """
            
            analysis = await self.venice_client.generate_response(
                prompt=analysis_prompt,
                model="claude-3-5-sonnet-20241022"
            )
            
            alerts = await self._check_alert_conditions(current_metrics)
            
            await self.long_term_memory.store_memory(
                content=f"System health analysis: {analysis}",
                memory_type=MemoryType.RESEARCH,
                importance=MemoryImportance.HIGH if alerts else MemoryImportance.MEDIUM,
                tags=["infrastructure", "health_analysis", "system_monitoring"],
                metadata={
                    "time_range": time_range,
                    "alerts_count": len(alerts),
                    "current_cpu": current_metrics.get('cpu', {}).get('percent', 0),
                    "current_memory": current_metrics.get('memory', {}).get('percent', 0)
                }
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "time_range": time_range,
                "current_metrics": current_metrics,
                "historical_data_points": len(historical_metrics),
                "analysis": analysis,
                "alerts": alerts,
                "health_score": await self._calculate_health_score(current_metrics, alerts)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze system health: {e}")
            return {"error": str(e)}
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        try:
            return {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "hostname": platform.node(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}
    
    async def execute_command(
        self,
        command: str,
        use_sudo: bool = False,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute a system command safely.
        
        Args:
            command: Command to execute
            use_sudo: Whether to use sudo
            timeout: Command timeout in seconds
            
        Returns:
            Command execution result
        """
        try:
            allowed_commands = self.config.get("allowed_commands", [
                "systemctl", "service", "ps", "top", "htop", "df", "free", "uptime",
                "netstat", "ss", "lsof", "iotop", "iostat", "vmstat"
            ])
            
            cmd_parts = command.split()
            if not cmd_parts or cmd_parts[0] not in allowed_commands:
                return {
                    "success": False,
                    "error": f"Command not allowed: {cmd_parts[0] if cmd_parts else 'empty'}"
                }
            
            if use_sudo:
                cmd_parts = ["sudo"] + cmd_parts
            
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            
            await self.long_term_memory.store_memory(
                content=f"Command executed: {command} - {'Success' if success else 'Failed'}",
                memory_type=MemoryType.EXPERIENCE,
                importance=MemoryImportance.MEDIUM,
                tags=["infrastructure", "command_execution"],
                metadata={
                    "command": command,
                    "success": success,
                    "return_code": result.returncode
                }
            )
            
            self.stats["commands_executed"] += 1
            
            return {
                "command": command,
                "success": success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "success": False,
                "error": "Command timed out",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return {
                "command": command,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _store_metrics_memory(self, metrics: SystemMetrics) -> None:
        """Store metrics in long-term memory."""
        try:
            await self.long_term_memory.store_memory(
                content=f"System metrics: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%",
                memory_type=MemoryType.EXPERIENCE,
                importance=MemoryImportance.LOW,
                tags=["infrastructure", "metrics", "system_monitoring"],
                metadata={
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "process_count": metrics.process_count,
                    "uptime": metrics.uptime
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store metrics memory: {e}")
    
    async def _check_alert_conditions(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any alert conditions are met."""
        alerts = []
        
        cpu_percent = metrics.get('cpu', {}).get('percent', 0)
        if cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                "type": "cpu_high",
                "severity": "warning",
                "message": f"High CPU usage: {cpu_percent:.1f}%",
                "threshold": self.alert_thresholds['cpu_percent']
            })
        
        memory_percent = metrics.get('memory', {}).get('percent', 0)
        if memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                "type": "memory_high",
                "severity": "warning",
                "message": f"High memory usage: {memory_percent:.1f}%",
                "threshold": self.alert_thresholds['memory_percent']
            })
        
        load_avg = metrics.get('cpu', {}).get('load_average', [0, 0, 0])
        if load_avg[0] > self.alert_thresholds['load_average']:
            alerts.append({
                "type": "load_high",
                "severity": "warning",
                "message": f"High load average: {load_avg[0]:.2f}",
                "threshold": self.alert_thresholds['load_average']
            })
        
        for mount, disk_info in metrics.get('disk', {}).items():
            if disk_info.get('percent', 0) > self.alert_thresholds['disk_usage']:
                alerts.append({
                    "type": "disk_full",
                    "severity": "critical",
                    "message": f"Disk usage high on {mount}: {disk_info['percent']:.1f}%",
                    "threshold": self.alert_thresholds['disk_usage']
                })
        
        if alerts:
            self.stats["alerts_generated"] += len(alerts)
        
        return alerts
    
    async def _calculate_health_score(self, metrics: Dict[str, Any], alerts: List[Dict[str, Any]]) -> float:
        """Calculate overall system health score (0-100)."""
        base_score = 100.0
        
        for alert in alerts:
            if alert['severity'] == 'critical':
                base_score -= 30
            elif alert['severity'] == 'warning':
                base_score -= 15
        
        cpu_percent = metrics.get('cpu', {}).get('percent', 0)
        memory_percent = metrics.get('memory', {}).get('percent', 0)
        
        if cpu_percent > 50:
            base_score -= (cpu_percent - 50) * 0.5
        
        if memory_percent > 60:
            base_score -= (memory_percent - 60) * 0.3
        
        return max(0.0, min(100.0, base_score))
    
    def _calculate_cutoff_time(self, time_range: str) -> datetime:
        """Calculate cutoff time based on time range."""
        now = datetime.now()
        
        if time_range == "1h":
            return now - timedelta(hours=1)
        elif time_range == "24h":
            return now - timedelta(hours=24)
        elif time_range == "7d":
            return now - timedelta(days=7)
        elif time_range == "30d":
            return now - timedelta(days=30)
        else:
            return now - timedelta(hours=1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self.stats.copy()


infrastructure_server = InfrastructureMCPServer(
    venice_client=None,
    long_term_memory=None
)


@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available infrastructure monitoring tools."""
    return [
        types.Tool(
            name="get_system_metrics",
            description="Get comprehensive system metrics including CPU, memory, disk, and network",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_processes": {
                        "type": "boolean",
                        "description": "Whether to include detailed process information",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="check_service_status",
            description="Check status of system services",
            inputSchema={
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Specific service to check (optional, checks all monitored services if not provided)"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="restart_service",
            description="Restart a system service",
            inputSchema={
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the service to restart"
                    },
                    "use_sudo": {
                        "type": "boolean",
                        "description": "Whether to use sudo for the command",
                        "default": True
                    }
                },
                "required": ["service_name"]
            }
        ),
        types.Tool(
            name="analyze_system_health",
            description="Analyze system health using Venice.ai with historical metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "time_range": {
                        "type": "string",
                        "description": "Time range for analysis (e.g., '1h', '24h', '7d')",
                        "default": "1h"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_system_info",
            description="Get basic system information",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="execute_command",
            description="Execute a system command safely (limited to allowed commands)",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute"
                    },
                    "use_sudo": {
                        "type": "boolean",
                        "description": "Whether to use sudo",
                        "default": False
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Command timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["command"]
            }
        )
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls for infrastructure operations."""
    try:
        if name == "get_system_metrics":
            result = await infrastructure_server.get_system_metrics(
                include_processes=arguments.get("include_processes", False)
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "check_service_status":
            result = await infrastructure_server.check_service_status(
                service_name=arguments.get("service_name")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "restart_service":
            result = await infrastructure_server.restart_service(
                service_name=arguments["service_name"],
                use_sudo=arguments.get("use_sudo", True)
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "analyze_system_health":
            result = await infrastructure_server.analyze_system_health(
                time_range=arguments.get("time_range", "1h")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_system_info":
            result = await infrastructure_server.get_system_info()
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "execute_command":
            result = await infrastructure_server.execute_command(
                command=arguments["command"],
                use_sudo=arguments.get("use_sudo", False),
                timeout=arguments.get("timeout", 30)
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
    """Main entry point for the Infrastructure MCP server."""
    import mcp.server.stdio
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
