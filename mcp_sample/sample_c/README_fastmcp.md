# FastMCP 服务器使用指南

这个项目展示了如何使用 FastMCP 创建一个功能丰富的 MCP (Model Context Protocol) 服务器。

## 📁 文件结构

```
mcp_sample/
├── fastmcp_server.py      # 主要的 FastMCP 服务器
├── test_fastmcp_client.py # 带 LLM 的测试客户端
├── simple_test.py         # 简单测试脚本
├── server.py              # 原始简单服务器
├── mcp_with_llm.py        # 原始 LLM 客户端
├── data/
│   └── kb.json           # 知识库数据
└── README_fastmcp.md     # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install fastmcp langchain-openai python-dotenv
```

### 2. 设置环境变量

创建 `.env` 文件：

```bash
API_KEY=your_openai_api_key
BASE_URL=your_openai_base_url  # 可选，如果使用自定义端点
```

### 3. 运行服务器

```bash
cd mcp_sample
python fastmcp_server.py
```

## 🛠️ 可用工具

FastMCP 服务器提供以下工具：

### 📚 知识库工具
- **`get_knowledge_base()`**: 获取完整的知识库内容
- **`search_knowledge_base(query: str)`**: 搜索知识库中的特定信息

### 💻 系统工具
- **`get_system_info()`**: 获取系统信息（OS、Python版本等）
- **`get_current_working_directory()`**: 获取当前工作目录

### 📁 文件操作工具
- **`list_directory_contents(path: str = ".")`**: 列出目录内容
- **`read_file_content(file_path: str, max_lines: int = 50)`**: 读取文件内容
- **`create_file(file_path: str, content: str)`**: 创建新文件
- **`get_file_info(file_path: str)`**: 获取文件详细信息

### 🧮 数学计算工具
- **`calculate_math(expression: str)`**: 执行安全的数学计算

## 🧪 测试方法

### 方法 1: 简单测试（推荐）

直接测试服务器功能，不需要 LLM：

```bash
python simple_test.py
```

这个脚本会：
- 连接到 FastMCP 服务器
- 列出所有可用工具
- 逐个测试每个工具的功能
- 显示详细的执行结果

### 方法 2: LLM 集成测试

使用 LLM 自动选择合适的工具：

```bash
python test_fastmcp_client.py
```

这个脚本会：
- 连接到 FastMCP 服务器
- 使用 LLM 分析用户查询
- 自动选择合适的工具
- 执行工具并总结结果

## 📖 使用示例

### 示例 1: 查询知识库

```python
# 获取所有知识库内容
result = await session.call_tool("get_knowledge_base", {})

# 搜索特定信息
result = await session.call_tool("search_knowledge_base", {"query": "vacation"})
```

### 示例 2: 文件操作

```python
# 列出当前目录内容
result = await session.call_tool("list_directory_contents", {"path": "."})

# 读取文件内容
result = await session.call_tool("read_file_content", {"file_path": "data/kb.json"})

# 创建新文件
result = await session.call_tool("create_file", {
    "file_path": "test.txt", 
    "content": "Hello World!"
})
```

### 示例 3: 数学计算

```python
# 基本计算
result = await session.call_tool("calculate_math", {"expression": "2 + 3 * 4"})

# 使用数学函数
result = await session.call_tool("calculate_math", {"expression": "sqrt(16) + pi"})
```

## 🔧 自定义和扩展

### 添加新工具

在 `fastmcp_server.py` 中添加新的工具函数：

```python
@mcp.tool()
def your_new_tool(param1: str, param2: int) -> str:
    """你的新工具描述
    
    Args:
        param1: 参数1描述
        param2: 参数2描述
    
    Returns:
        返回结果描述
    """
    try:
        # 你的工具逻辑
        result = f"处理结果: {param1}, {param2}"
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"
```

### 修改服务器配置

```python
# 修改服务器名称和端口
mcp = FastMCP(
    name="Your Custom Server",
    host="0.0.0.0",
    port=8080,  # 自定义端口
)
```

## 🚨 安全注意事项

1. **文件操作**: 服务器可以读取和创建文件，确保在生产环境中限制访问权限
2. **数学计算**: 使用了安全的 `eval()` 实现，只允许预定义的函数和常量
3. **路径验证**: 所有文件操作都使用 `Path.resolve()` 来防止路径遍历攻击

## 🔍 故障排除

### 常见问题

1. **导入错误**: 确保安装了所有依赖包
2. **权限错误**: 检查文件读写权限
3. **连接失败**: 确保服务器脚本路径正确

### 调试模式

在服务器代码中添加调试信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 相关资源

- [FastMCP 文档](https://gofastmcp.com/)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [LangChain 文档](https://python.langchain.com/)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

---

**注意**: 这是一个演示项目，展示了 FastMCP 的基本用法。在生产环境中使用时，请根据具体需求调整安全设置和功能限制。 