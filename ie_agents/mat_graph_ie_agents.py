from autogen import ConversableAgent
from dotenv import load_dotenv
import os, csv, json, re

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
llm_config = {
    "model": "gpt-4o",
    "api_key": api_key,
    "base_url": base_url,
    "temperature": 0.5,
    "cache_seed": 13   # 类似的问题，不会再次请求，而是去到缓存中查询，并返回结果
}

# 分类路径嵌套结构（供泛化与分类 Agent 使用）
classification_path_ontology = """
你需要参考以下【材料科学分类体系】：

分类路径 = {
  "object": {
    "材料": ["功能材料", "衬底材料", "过程材料", "结构材料"],
    "超材料": [],
    "器件": {
      "组件": ["结构单元"]
    }
  },
  "property": {
    "材料特性": [
      "材料热学特性", "材料光学特性", "材料电学特性", "材料电磁特性",
      "材料磁学特性", "材料传感特性", "材料机械特性", "材料化学特性",
      "材料声学特性", "材料表面与界面特性", "材料响应特性",
      "材料生物相容性", "材料加工与工艺特性"
    ],
    "器件特性": [
      "器件特性、指标", "器件结构特性、指标", "组件特性、指标", "结构单元特性、指标"
    ]
  },
  "现象、原理、机理、物理效应": [],
  "物理场、激励": [],
  "化学场、激励": [],
  "科学难题": [],
  "技术难题": [],
  "可实现实体": {
    "应用场景": [],
    "角色": [],
    "实现功能": []
  },
  "process": {
    "材料制备": [
      "物理合成与改性技术", "化学合成与改性技术", "增材制造技术", "表面与结构工程"
    ],
    "器件制备": ["微纳加工", "集成组装"],
    "材料表征、测量": ["物性测量", "尺寸测量"],
    "器件测量": ["器件性能测量"],
    "计算、仿真": [],
    "设计": []
  },
  "processing_node": {
    "材料表征设备": ["材料表征设备部分、组件"],
    "材料制备设备": ["材料制备设备部分、组件"],
    "器件表征设备": ["器件表征设备部分、组件"],
    "器件制备设备": ["器件制备设备部分、组件"],
    "仿真、计算软件": [],
    "仿真、计算设备": []
  }
}
"""
    
def build_agent(name, prompt, inject=False):
    if inject:
        full_prompt = f"{prompt.strip()}\n\n参考分类体系结构：\n{classification_path_ontology}"
    else:
        full_prompt = prompt.strip()
    
    agent = ConversableAgent(
        name=name,
        system_message=full_prompt,  # 分类体系在这里
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    return agent

# 1. 实体抽取代理
entity_extractor = build_agent(
    "EntityExtractionAgent", 
    """
    你是一位精通材料科学的实体识别专家。
    请根据以下文献摘要，识别出所有与材料科学分类体系相关的实体，保持其原文形式（包括单位，不添加说明词）。
    - 所有实体必须与常见材料领域分类语义关联
    - 可识别的实体包括：材料、器件、结构细节、数值+单位、性能行为、物理/化学/光学/热学效应、测量环境、制备工艺、技术难题、应用场景、设备等
    - 不应抽取宽泛或无实际指代词（如"结构"、"多个材料"）。
    - 输出为实体原文字符串数组（允许重复），如：[实体1, 实体2, ...]
    """
)

# 3. 概念泛化代理
generalizer = build_agent(
    "GeneralizationAgent",
    """
    你是一位材料科学知识工程专家。
    任务是：对输入实体列表，根据其上下文，为每个实体生成1到多条泛化概念路径。
    请使用如下路径模板作为参考：[具体术语, 中级抽象, 上层类别, 顶层分类]
    例如：
    "585 THz" → ["共振频率", "光学频率", "测量参数", "物理场"]
    "金属钨" → ["金属钨", "功能材料", "材料"]
    - 输入格式：[{"entity": "...", "context": "..."}, {"entity": "...", "context": "..."}, ...]
    - 输出格式：{"实体原文1": [[路径1], [路径2], ...], "实体原文2": [[路径1], [路径2], ...], ...}
    - 每个路径为一组泛化概念层级（1~5级）
    - 若仅有一种泛化路径，也以数组形式返回
    - 注意：泛化数组均为中文。
    """,
    inject=True
)

# 4. 实体分类代理
classifier = build_agent(
    "ClassificationAgent",
    """
    你是一位材料本体分类专家。

    请根据输入实体的上下文和泛化路径，在【材料科学分类体系】中为每条泛化路径推荐最贴近的末端分类节点。

    **分类逻辑要求如下：**
    1. 若某路径可准确对应分类体系中的末端分类，请直接填写该末端分类至 classification 字段，**并将 suggestion 设为空字符串**。
    2. 若路径无法匹配具体分类节点，请将 classification 设为"其他"，并在 suggestion 中写明建议的分类节点及其理由，格式为：“建议为：材料电磁特性，因为...”。

    【输入格式】
    [
    {
        "entity": "xxx",
        "context": "...",
        "generalization": [["...", "..."]]
    },
    ...
    ]

    【输出格式】
    {
    "实体名1": [
        {
            "generalization": ["...", "..."],
            "classification": "xxx",
            "suggestion": "..."
        }
    ],
    ...
    }

    注意：
    - classification 和 suggestion 必须是中文。
    - 一个实体可能有多个上下文，每个上下文可有多个路径和分类。
    """,
    inject=True
)

def call_agent(agent, input_text):
    response = agent.generate_reply(messages=[{"role": "user", "content": input_text}])
    response = re.sub(r"^```json\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(response)
    except:
        print(type(response))
        print("Raw Response repr:\n", repr(response))
        print("解析失败...")

# 从上下文中定位局部窗口（关键词匹配 + 语义窗口）
def extract_local_context(entity, full_text, window_size=50):
    pattern = re.compile(re.escape(entity))
    matches = list(pattern.finditer(full_text))
    contexts = set()
    for m in matches:
        start = max(0, m.start() - window_size)
        end = min(len(full_text), m.end() + window_size)
        context = full_text[start:end].strip()
        contexts.add(context)
    return list(contexts)

def process_abstract(abs_id, title, abstract, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    text = f"{title}\n{abstract}"
    try:
        raw_entities = call_agent(entity_extractor, text)
        unique_entities = list(set(raw_entities))
    except:
        print(f"实体抽取失败： {abs_id}")
        return
        
    # 批量收集所有实体的上下文
    batch_entities = []
    for entity in unique_entities:
        local_contexts = extract_local_context(entity, text)
        for context in local_contexts:
            batch_entities.append({"entity": entity, "context": context})
            
    print(f"总实体数量: {len(batch_entities)}")
    
    # 检查输入长度，如果太长则分批处理
    batch_size = 10  # 每批处理10个实体
    all_gen_results = {}
    
    for i in range(0, len(batch_entities), batch_size):
        batch = batch_entities[i:i+batch_size]
        print(f"处理批次 {i//batch_size + 1}: {len(batch)} 个实体")
        
        try:
            batch_gen_input = json.dumps(batch, ensure_ascii=False)
            batch_gen_result = call_agent(generalizer, batch_gen_input)
            all_gen_results.update(batch_gen_result)
        except Exception as e:
            print(f"批次 {i//batch_size + 1} 泛化失败： {e}")
            continue
    
    if not all_gen_results:
        print(f"所有泛化处理都失败了： {abs_id}")
        return
    
    # 批量分类处理
    batch_cls_input = []
    for item in batch_entities:
        entity = item["entity"]
        context = item["context"]
        gen_paths = all_gen_results.get(entity, [])
        batch_cls_input.append({
            "entity": entity,
            "context": context,
            "generalization": gen_paths
        })
        
    # 分批处理分类
    all_cls_results = {}
    for i in range(0, len(batch_cls_input), batch_size):
        batch = batch_cls_input[i:i+batch_size]
        print(f"分类处理批次 {i//batch_size + 1}: {len(batch)} 个实体")
        
        try:
            batch_cls_json = json.dumps(batch, ensure_ascii=False)
            batch_cls_result = call_agent(classifier, batch_cls_json)
            all_cls_results.update(batch_cls_result)
        except Exception as e:
            print(f"分类批次 {i//batch_size + 1} 失败： {e}")
            continue
    
    if not all_cls_results:
        print(f"所有分类处理都失败了： {abs_id}")
        return
        
    # 整理结果
    result = {}
    for item in batch_entities:
        entity = item["entity"]
        context = item["context"]
        gen_paths = all_gen_results.get(entity, [])
        cls_results = all_cls_results.get(entity, [])

        for i, path in enumerate(gen_paths):
            # 尝试找到对应分类结果
            cls_info = cls_results[i] if i < len(cls_results) else {"classification": "其他", "suggestion": ""}

            classification = cls_info.get("classification", "")
            suggestion = cls_info.get("suggestion", "")

            if classification != "其他":
                suggestion = ""
            elif classification == "其他" and suggestion:
                match = re.match(r"建议为[:：]?(.*?)[，, 。]", suggestion.strip())
                if match:
                    classification = match.group(1).strip()
                    suggestion = ""

            # 构建目标结构
            record = {
                "context": context,
                "generalization": path,
                "classification": classification,
                "suggestion": suggestion
            }

            if entity not in result:
                result[entity] = []
            result[entity].append(record)
    
    with open(os.path.join(output_dir, f"{abs_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
import time
def run_pipeline_from_csv(csv_path, output_dir="outputs"):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ts = time.time()
            print(f"---------------------- 正在处理 {row['id']} ----------------------")
            process_abstract(row["id"], row["title"], row["abstract"], output_dir)
            print(time.time() - ts)
            
            
run_pipeline_from_csv(csv_path="samples.csv")