# 工具服务器，通过 FastMCP 注册工具，并提供给客户端调用

from mcp.server.fastmcp import FastMCP
import requests

print("[server] script started")

# 创建 FastMCP 实例，命名为 multi_tool_server
mcp = FastMCP("multi_tool_server")

# 注册天气查询工具，可被客户端远程调用
@mcp.tool()
def get_weather(location: str, days: int = 1) -> dict:
    """
    查询指定地点的天气情况
    Args:
        location (str): 地点名称
        days (int, optional): 查询天数. Defaults to 1.
    Returns:
        dict: 天气数据
    """
    print(f"[server] get_weather called with: {location}, {days}")
    geo_resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": location, "format": "json", "limit": 1},
        headers={"User-Agent": "mcp-sample-app"}
    )
    geo_data = geo_resp.json()
    if not geo_data:
        return {"error": "No location found"}
    latitude = geo_data[0]["lat"]
    longitude = geo_data[0]["lon"]
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": latitude, 
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "Asia/Shanghai",
            "forecast_days": days
        })
    return resp.json()

# 注册书籍查询工具，可被客户端远程调用
@mcp.tool()
def find_book(title: str) -> dict:
    """
    查询指定书籍的详细信息
    Args:
        title (str): 书籍名称
    Returns:
        dict: 书籍信息
    """
    print(f"[server] find_book called with: {title}")
    resp = requests.get("https://openlibrary.org/search.json", params={"title": title})
    return resp.json()

if __name__ == "__main__":
    print("[server] about to run")
    try:
        # 启动 MCP 工具服务器，等待客户端连接和调用
        mcp.run()
    except Exception as e:
        print("[server] exception:", e)