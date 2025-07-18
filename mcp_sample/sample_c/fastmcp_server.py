import os
import json
import math
import platform
import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP(
    name="Enhanced Knowledge Base & Tools",
    host="0.0.0.0",
    port=8050,
)


@mcp.tool()
def get_knowledge_base() -> str:
    """Retrieve the entire knowledge base as a formatted string.
    
    Returns:
        A formatted string containing all Q&A pairs from the knowledge base.
    """
    try:
        kb_path = os.path.join(os.path.dirname(__file__), "data", "kb.json")
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)

        kb_text = "📚 Knowledge Base Contents:\n\n"

        if isinstance(kb_data, list):
            for i, item in enumerate(kb_data, 1):
                if isinstance(item, dict):
                    question = item.get("question", "Unknown question")
                    answer = item.get("answer", "Unknown answer")
                else:
                    question = f"Item {i}"
                    answer = str(item)

                kb_text += f"❓ Q{i}: {question}\n"
                kb_text += f"💡 A{i}: {answer}\n\n"
        else:
            kb_text += f"Knowledge base content: {json.dumps(kb_data, indent=2)}\n\n"

        return kb_text
    except FileNotFoundError:
        return "❌ Error: Knowledge base file not found"
    except json.JSONDecodeError:
        return "❌ Error: Invalid JSON in knowledge base file"
    except Exception as e:
        return f"❌ Error: {str(e)}"


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information.
    
    Args:
        query: The search query to look for in questions and answers.
    
    Returns:
        Matching Q&A pairs from the knowledge base.
    """
    try:
        kb_path = os.path.join(os.path.dirname(__file__), "data", "kb.json")
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)

        if not isinstance(kb_data, list):
            return "❌ Error: Knowledge base format is not a list"

        query_lower = query.lower()
        matches = []

        for item in kb_data:
            if isinstance(item, dict):
                question = item.get("question", "").lower()
                answer = item.get("answer", "").lower()
                
                if query_lower in question or query_lower in answer:
                    matches.append(item)

        if matches:
            result = f"🔍 Search results for '{query}':\n\n"
            for i, match in enumerate(matches, 1):
                result += f"📋 Result {i}:\n"
                result += f"❓ Q: {match.get('question', 'Unknown')}\n"
                result += f"💡 A: {match.get('answer', 'Unknown')}\n\n"
            return result
        else:
            return f"🔍 No matches found for '{query}' in the knowledge base."

    except Exception as e:
        return f"❌ Error searching knowledge base: {str(e)}"


@mcp.tool()
def get_system_info() -> str:
    """Get current system information including OS, Python version, and platform details.
    
    Returns:
        System information as a formatted string.
    """
    try:
        info = {
            "OS": platform.system(),
            "OS Version": platform.version(),
            "Architecture": platform.machine(),
            "Python Version": platform.python_version(),
            "Current Directory": os.getcwd(),
            "Current Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "User": os.getenv("USER", "Unknown"),
            "Home Directory": os.path.expanduser("~")
        }
        
        result = "🖥️ System Information:\n\n"
        for key, value in info.items():
            result += f"📌 {key}: {value}\n"
        
        return result
    except Exception as e:
        return f"❌ Error getting system info: {str(e)}"


@mcp.tool()
def list_directory_contents(path: str = ".") -> str:
    """List contents of a directory.
    
    Args:
        path: Directory path to list (default: current directory).
    
    Returns:
        Directory contents as a formatted string.
    """
    try:
        dir_path = Path(path).resolve()
        
        if not dir_path.exists():
            return f"❌ Error: Directory '{path}' does not exist"
        
        if not dir_path.is_dir():
            return f"❌ Error: '{path}' is not a directory"
        
        result = f"📁 Directory contents of '{dir_path}':\n\n"
        
        # List directories first
        dirs = [item for item in dir_path.iterdir() if item.is_dir()]
        if dirs:
            result += "📂 Directories:\n"
            for d in sorted(dirs):
                result += f"  - {d.name}/\n"
            result += "\n"
        
        # List files
        files = [item for item in dir_path.iterdir() if item.is_file()]
        if files:
            result += "📄 Files:\n"
            for f in sorted(files):
                size = f.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                result += f"  - {f.name} ({size_str})\n"
        
        if not dirs and not files:
            result += "📭 Directory is empty"
        
        return result
    except Exception as e:
        return f"❌ Error listing directory: {str(e)}"


@mcp.tool()
def read_file_content(file_path: str, max_lines: int = 50) -> str:
    """Read and display the content of a file.
    
    Args:
        file_path: Path to the file to read.
        max_lines: Maximum number of lines to display (default: 50).
    
    Returns:
        File content as a string.
    """
    try:
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            return f"❌ Error: File '{file_path}' does not exist"
        
        if not file_path.is_file():
            return f"❌ Error: '{file_path}' is not a file"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        display_lines = lines[:max_lines]
        
        result = f"📖 File: {file_path}\n"
        result += f"📊 Total lines: {total_lines}\n"
        result += f"📋 Showing first {len(display_lines)} lines:\n\n"
        
        for i, line in enumerate(display_lines, 1):
            result += f"{i:3d}: {line.rstrip()}\n"
        
        if total_lines > max_lines:
            result += f"\n... and {total_lines - max_lines} more lines"
        
        return result
    except UnicodeDecodeError:
        return f"❌ Error: File '{file_path}' contains binary data or unsupported encoding"
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"


@mcp.tool()
def calculate_math(expression: str) -> str:
    """Perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4").
    
    Returns:
        Result of the calculation.
    """
    try:
        # Define allowed functions and constants
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'sqrt': math.sqrt,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'log10': math.log10, 'exp': math.exp,
            'pi': math.pi, 'e': math.e
        }
        
        # Compile and evaluate the expression
        compiled = compile(expression, '<string>', 'eval')
        
        # Check for disallowed names
        for name in compiled.co_names:
            if name not in allowed_names:
                return f"❌ Error: '{name}' is not allowed in mathematical expressions"
        
        result = eval(compiled, {"__builtins__": {}}, allowed_names)
        
        return f"🧮 Calculation: {expression} = {result}"
    except ZeroDivisionError:
        return "❌ Error: Division by zero"
    except ValueError as e:
        return f"❌ Error: Invalid mathematical operation - {str(e)}"
    except Exception as e:
        return f"❌ Error evaluating expression: {str(e)}"


@mcp.tool()
def create_file(file_path: str, content: str) -> str:
    """Create a new file with specified content.
    
    Args:
        file_path: Path where the file should be created.
        content: Content to write to the file.
    
    Returns:
        Status message about file creation.
    """
    try:
        file_path = Path(file_path)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✅ File '{file_path}' created successfully with {len(content)} characters"
    except Exception as e:
        return f"❌ Error creating file: {str(e)}"


@mcp.tool()
def get_current_working_directory() -> str:
    """Get the current working directory.
    
    Returns:
        Current working directory path.
    """
    try:
        cwd = os.getcwd()
        return f"📂 Current working directory: {cwd}"
    except Exception as e:
        return f"❌ Error getting current directory: {str(e)}"


@mcp.tool()
def get_file_info(file_path: str) -> str:
    """Get detailed information about a file.
    
    Args:
        file_path: Path to the file.
    
    Returns:
        File information as a formatted string.
    """
    try:
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            return f"❌ Error: File '{file_path}' does not exist"
        
        stat = file_path.stat()
        
        # Get file type
        if file_path.is_file():
            file_type = "File"
        elif file_path.is_dir():
            file_type = "Directory"
        elif file_path.is_symlink():
            file_type = "Symbolic Link"
        else:
            file_type = "Other"
        
        # Format file size
        size = stat.st_size
        if size < 1024:
            size_str = f"{size} bytes"
        elif size < 1024 * 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size/(1024*1024):.1f} MB"
        
        # Get modification time
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        
        result = f"📄 File Information: {file_path}\n\n"
        result += f"📌 Type: {file_type}\n"
        result += f"📏 Size: {size_str}\n"
        result += f"🕒 Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"🔐 Permissions: {oct(stat.st_mode)[-3:]}\n"
        
        return result
    except Exception as e:
        return f"❌ Error getting file info: {str(e)}"


# Run the server
if __name__ == "__main__":
    print("🚀 Starting Enhanced FastMCP Server...")
    print("📋 Available tools:")
    print("  - get_knowledge_base: Retrieve all knowledge base content")
    print("  - search_knowledge_base: Search for specific information")
    print("  - get_system_info: Get system information")
    print("  - list_directory_contents: List directory contents")
    print("  - read_file_content: Read file contents")
    print("  - calculate_math: Perform mathematical calculations")
    print("  - create_file: Create new files")
    print("  - get_current_working_directory: Get current directory")
    print("  - get_file_info: Get detailed file information")
    print("\n🔧 Server running on stdio transport...")
    
    mcp.run(transport="stdio") 