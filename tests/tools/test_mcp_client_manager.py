import asyncio
from types import SimpleNamespace

import pytest


class FakeClient:
    def __init__(self, tool):
        self.tool = tool

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def list_tools_mcp(self):
        return SimpleNamespace(tools=[self.tool])


def test_duplicate_tool_names_across_mcp_servers_fail_explicitly():
    pytest.importorskip("fastmcp")
    from verl.tools.utils.mcp_clients.McpClientManager import MCPClientManager

    def search_tool(description):
        return SimpleNamespace(
            name="search",
            description=description,
            inputSchema={"type": "object", "properties": {}},
        )

    manager = MCPClientManager()
    manager.clients = [FakeClient(search_tool("server one")), FakeClient(search_tool("server two"))]
    manager.tool_client_mapping = {}

    with pytest.raises(ValueError, match="Duplicate MCP tool name 'search'"):
        asyncio.run(manager.fetch_tool_schemas(tool_selected_list=[]))

    assert manager.tool_client_mapping == {}
