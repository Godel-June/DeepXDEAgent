import os


# 动态生成提示词（避免硬编码）
def build_difficulty_prompt(question):
    return f"""
Given the following differential equation:  
{question}  

Please evaluate the difficulty of solving it with DeepXDE based on these criteria:  

1) Easy:  
    - The equation is linear, low-dimensional (1D/2D), and has standard boundary conditions.  
    - Can be solved directly using default DeepXDE configurations without additional tuning.  

2) Medium:  
    - The equation contains nonlinear terms, moderate-dimensional domains (3D), or complex boundary conditions (e.g., time-dependent, discontinuous).  
    - Requires parameter tuning, custom loss functions, or retrieval of similar solutions (RAG) for optimization.  

3) Difficult:  
    - The equation is high-dimensional (4D+), involves multi-physics coupling, or requires adaptive methods (e.g., PINNs with domain decomposition).  
    - Needs collaborative workflows (e.g., multi-model integration, hybrid numerical-PINN approaches) or advanced techniques.  

Provide the difficulty level [1) Easy, 2) Medium, or 3) Difficult] and a brief justification.
    """
    
def build_query_split_prompt(user_input: str, tools_info_dict: dict) -> str:
    prompt = f"""
你是一个专注于求解PDE问题的智能助手，能够利用deepxde库来求解PDE问题。
我会提供给你一系列用deepxde库写好的python函数作为待调用的工具，你需要根据用户输入的PDE求解问题和你对PDE求解问题的理解，生成如何调用这些工具函数来求解PDE问题的具体步骤。
请注意：
- 整体求解步骤符合利用deepxde库求解PDE问题的一般流程，保证顺序执行这些步骤可以求解用户的PDE问题。
- 每一步骤应当包含你的思考过程，即选择工具的理由，以及所选择的工具名称
- 每一个步骤只能调用一个工具函数；一般来说，一个工具函数不会被多次调用
- 用户输入中的数学表达式都使用LaTex格式，你的输出如果需要包含数学表达式，也需要使用LaTex格式


所提供的工具函数即对应功能如下：
    - define_pde():{tools_info_dict.get("define_pde", None)}
    - define_reference_solution():{tools_info_dict.get("define_reference_solution", None)}
    - define_domain()：{tools_info_dict.get("define_domain", None)}
    - define_initial_condition()：{tools_info_dict.get("define_initial_condition", None)}
    - define_boundary_condition()：{tools_info_dict.get("define_boundary_condition", None)}
    - create_training_data()：{tools_info_dict.get("create_training_data", None)}
    - create_network()：{tools_info_dict.get("create_network", None)}
    - train_model()：{tools_info_dict.get("train_model", None)}
    - train_model_LBFGS()：{tools_info_dict.get("train_model_LBFGS", None)}
    - visualize_and_save()：{tools_info_dict.get("visualize_and_save", None)}

你的输出中应包含思考过程和具体答案，具体答案请严格依照以下JSON格式，确保可以使用json.loads()来解析：
[
{{"task_id": "1", "tool_name": "所选工具名称", "reasoning": "选择此工具的理由"}},
{{"task_id": "2", "tool_name": "所选工具名称", "reasoning": "选择此工具的理由"}},
...
]
其中，task_id为任务编号，从1开始，tool_name为工具名称，reasoning为选择此工具的理由，理由中不要使用公式，用纯文字来描述。

请注意：
- 输出的答案部分不要包含数学公式
- 如果JSON字符串中包含换行符 (\n)、制表符 (\t) 等控制字符，需要将它们转义为 \\n 或 \\t。
- 确保可以使用python中的json.loads()方法解析为列表格式。

示例：
# 用户输入
我想求解如下扩散方程：\\frac{{\\partial y}}{{\\partial t}} = \\frac{{\\partial^2y}}{{\\partial x^2}} - e^{{-t}}(\\sin(\pi x) - \\pi^2\\sin(\\pi x))
其中，\\qquad x \\in [-1, 1], \\quad t \\in [0, 1]
初始条件y(x, 0) = \\sin(\\pi x)
迪利克雷边界条件y(-1, t) = y(1, t) = 0
参考解为y = e^{{-t}} \\sin(\\pi x)

# 你的输出
<think>
思考过程的具体内容
</think>
[
{{"task_id": "1", "tool_name": "define_pde", "reasoning": "根据扩散方程的形式，选择define_pde函数来定义PDE方程"}},
{{"task_id": "2", "tool_name": "define_reference_solution", "reasoning": "由于提供了参考解，选择define_reference_solution函数定义参考解"}},
{{"task_id": "3", "tool_name": "define_domain", "reasoning": "根据提供的区域和时间区间，选择define_domain函数定义求解区域和时间区间"}},
{{"task_id": "4", "tool_name": "define_initial_condition", "reasoning": "根据提供的初始条件，选择define_initial_condition函数定义初始条件"}},
{{"task_id": "5", "tool_name": "define_boundary_condition", "reasoning": "根据提供的边界条件，选择define_boundary_condition函数定义边界条件"}},
{{"task_id": "6", "tool_name": "create_training_data", "reasoning": "创建训练数据"}},
{{"task_id": "7", "tool_name": "create_network", "reasoning": "创建网络结构"}},
{{"task_id": "8", "tool_name": "train_model", "reasoning": "先使用Adam优化器训练模型"}},
{{"task_id": "9", "tool_name": "train_model_LBFGS", "reasoning": "由于问题比较复杂，利用LBFGS优化器进一步训练模型"}},
{{"task_id": "10", "tool_name": "visualize_and_save", "reasoning": "训练完成后，可视化训练过程并保存训练历史"}},
]

根据以上示例，处理下面的用户输入：
{user_input}
"""
    return prompt

def determine_tool_params_prompt(tool_info, previous_steps_info):
    # 得到工具的参数信息
    tool_params_prompt = "\n".join(
        f"参数{i + 1}:\n    参数名称：{key}\n    参数类型：{value['type']}\n    参数描述：{value['description']}\n"
        for i, (key, value) in enumerate(tool_info[1]["parameters"].items())
    )
    prompt = f"""
### 目前所需调用工具的具体信息如下：
工具名称：{tool_info[0]}
功能描述：{tool_info[1]["description"]}
所需参数的名称、类型和描述：
{tool_params_prompt}

### 之前的工具执行结果如下，包括每一步骤调用工具的名称、参数信息和返回结果(结果类型以及具体的值)，需要注意的是当工具返回结果的类型是list时，它的类型信息是用来[第一个元素的类型，第二个元素的类型......]表示的：
{previous_steps_info}

请你根据所有相关信息确定当前调用工具的参数，输出一个json对象，确保它可以通过json.loads()转换为python字典。
回答时，请注意：
  - 你应该先详细展示你的思考过程，以思考开始标签<think>和思考结束标签</think>标识；
  - 思考结束标签</think>后输出具体答案内容，注意换行。
"""
    return prompt

    
if __name__ == '__main__':
    print(build_difficulty_prompt("test_problem"))