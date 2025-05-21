import os
import json


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
You are an intelligent assistant focused on solving PDE problems using the DeepXDE library.
I will provide you with a series of Python functions written using the DeepXDE library as tools to solve PDE problems.
You need to generate a step-by-step procedure for calling these tools to solve the PDE problem based on the user's input and your understanding of the PDE problem.
Please note:
- The overall solution process follows the general process of using the DeepXDE library to solve PDE problems, ensuring that the steps are executed in order to solve the user's PDE problem.
- Each step should include your thinking process, i.e., the reason for selecting the tool and the name of the selected tool.
- Each step can only call one tool function; generally, a tool function should not be called multiple times.
- The mathematical expressions in the user's input are in LaTex format, and your output should also use LaTex format if it includes mathematical expressions.

The provided tool functions correspond to the following functionalities:
    - define_pde(): {tools_info_dict.get("define_pde", None)}
    - define_reference_solution():{tools_info_dict.get("define_reference_solution", None)}
    - define_domain(): {tools_info_dict.get("define_domain", None)}
    - define_initial_condition(): {tools_info_dict.get("define_initial_condition", None)}
    - define_boundary_condition(): {tools_info_dict.get("define_boundary_condition", None)}
    - create_training_data(): {tools_info_dict.get("create_training_data", None)}
    - create_network(): {tools_info_dict.get("create_network", None)}
    - train_model(): {tools_info_dict.get("train_model", None)}
    - train_model_LBFGS(): {tools_info_dict.get("train_model_LBFGS", None)}
    - visualize_and_save(): {tools_info_dict.get("visualize_and_save", None)}

Your output should contain the thinking process and the specific answer, in the following JSON format, ensuring that it can be parsed by json.loads():
[
{{"task_id": "1", "tool_name": "Selected tool name", "reasoning": "Reason for selecting the tool"}},
{{"task_id": "2", "tool_name": "Selected tool name", "reasoning": "Reason for selecting the tool"}},
...
]
where, task_id is the task number, starting from 1, tool_name is the tool name, and reasoning is the reason for selecting the tool. The reasoning should not use mathematical formulas, but should be described in plain text.

Please note:
- The answer part of the output should not contain mathematical formulas.
- If the JSON string contains newline characters (\n) or tab characters (\t), they need to be escaped as \\n or \\t.
- Make sure that the output can be parsed as a list using the python json.loads() method.

Example:
# User input
I want to solve the following diffusion equation: \\frac{{\\partial y}}{{\\partial t}} = \\frac{{\\partial^2y}}{{\\partial x^2}} - e^{{-t}}(\\sin(\pi x) - \\pi^2\\sin(\\pi x))
where, \\qquad x \\in [-1, 1], \\quad t \\in [0, 1]
Initial condition y(x, 0) = \\sin(\\pi x)
Dirichlet boundary condition y(-1, t) = y(1, t) = 0
Reference solution y = e^{{-t}} \\sin(\\pi x)

# Your output
<think>
The thinking process for the solution process is as follows:
</think>
[
{{"task_id": "1", "tool_name": "define_pde", "reasoning": "According to the differential equation's form, we can use define_pde function to define the PDE equation."}},
{{"task_id": "2", "tool_name": "define_reference_solution", "reasoning": "The reference solution is provided, so we can use define_reference_solution function to define the reference solution."}},
{{"task_id": "3", "tool_name": "define_domain", "reasoning": "The domain and time interval are provided, so we can use define_domain function to define the domain and time interval."}},
{{"task_id": "4", "tool_name": "define_initial_condition", "reasoning": "The initial condition is provided, so we can use define_initial_condition function to define the initial condition."}},
{{"task_id": "5", "tool_name": "define_boundary_condition", "reasoning": "The boundary condition is provided, so we can use define_boundary_condition function to define the boundary condition."}},
{{"task_id": "6", "tool_name": "create_training_data", "reasoning": "We need to create training data."}},
{{"task_id": "7", "tool_name": "create_network", "reasoning": "We need to create a network structure."}},
{{"task_id": "8", "tool_name": "train_model", "reasoning": "We can use Adam optimizer to train the model."}},
{{"task_id": "9", "tool_name": "train_model_LBFGS", "reasoning": "The problem is complex, so we can use LBFGS optimizer to further train the model."}},
{{"task_id": "10", "tool_name": "visualize_and_save", "reasoning": "After the training is complete, we can visualize the training process and save the training history."}}
]

According to the above example, process the following user input:
{user_input}
"""
    return prompt

def determine_tool_params_prompt(tool_info, previous_steps_info):
    # 得到工具的参数信息
    tool_params_prompt = "\n".join(
        f"参数{i + 1}:\n\tParameter name: {key}\n\tParameter type: {value['type']}\n\tParameter description: {value['description']}\n"
        for i, (key, value) in enumerate(tool_info[1]["parameters"].items())
    )
    prompt = f"""
### The specific information of the tool you need to call is as follows:
Tool name: {tool_info[0]}
Function description: {tool_info[1]["description"]}
The required parameters' name, type, and description:
{tool_params_prompt}

### The previous tool execution results are as follows, including the name of the tool called at each step, parameter information, and the returned result (the type of the result and the specific value):
{previous_steps_info}

Please provide the parameter values for the tool you need to call, and output a json object, ensuring that it can be parsed by json.loads().
Please note:
  - You should first provide a detailed explanation of your thinking process, using the start tag <think> and end tag </think> to identify the thinking process;
  - Followed the end tag </think>, provide the specific answer content and pay attention to the line breaks.
"""
    return prompt

def redetermine_tool_params_prompt(tool_info, previous_steps_info, determine_continue_result):
    # 得到
    determine_tool_params_prompt = determine_tool_params_prompt(tool_info, previous_steps_info)
    prompt = f"""
After validation, PDE Orchestrator (pde_orchestrator) has identified an issue with the parameters for tool invocation or tool execution results. 
And it provides the reasoning: 
{determine_continue_result['reasoning']}

Please re-evaluate based on this reasoning and provide revised answers.
{determine_tool_params_prompt}
"""


# 生成转化标准格式提示词
def build_standard_format_prompt(problem_description):
    PROMPT_Standard_Format: str = """
Convert the following PDE problem into the following standard format:
PDE problem:
{PDE_problem}

Standard format:
{Standard_format}

Example:
{Example}

Formatting notes:
- If the parameter string is in LaTex format, you need to use LaTex format in the output string, and you should use (\frac) instead of "\\frac" in your output to avoid json.loads() parsing errors.
- If the JSON string contains control characters such as line breaks (\n) and tabs (\t), you need to escape them as \\n or \\t.
- The answer should be in the start tag <answer> and end tag </answer>
"""
    with open(os.path.join(os.path.dirname(__file__), "exp.json"), "r", encoding="utf-8") as f:
        exp_json = json.load(f)
        
    prompt = PROMPT_Standard_Format.format(
        PDE_problem=problem_description, 
        Standard_format=exp_json["Standard_format"], 
        Example=exp_json["Example"])
    return prompt

def build_retrieval_prompt(user_input: str) -> str:
    PROMPT_Find: str = """
Find the PDE problem case that most closely matches the following case:
{user_input}
Please note: where the specific type of PDE formula (if given) should be matched with the highest priority
"""
    prompt = PROMPT_Find.format(user_input=user_input)
    return prompt

def add_retrieval_case_to_prompt(question: str, retrieval_case: str, tools_info_dict: dict) -> str:
    init_prompt = build_query_split_prompt(question, tools_info_dict)
    # retrieval_case 是一个字典
    case_example = """Example:
# User input
{retrieval_problem}

# Your output
<think>
The thinking process for the solution process is as follows:
</think>
[
{task_plan}
]
"""
    # 构建 task_plan
    task_plan = ""
    for task in retrieval_case['expected_tool_chain']:
        task_plan += f"""{{"task_id": "{task['step']}", "tool_name": "{task['tool_name']}", "reasoning": "when answer, you should provide the reasoning for selecting the tool"}},\n"""
    task_plan = task_plan.strip(",\n")
    
    retrieval_case_plan = case_example.format(retrieval_problem=retrieval_case['problem_description'], task_plan=task_plan)
    
    PROMPT_Add_Retrieval_Case: str = """
{prompt}

We also found the following case for reference that closely matches the given PDE problem:
{retrieval_case_plan}
"""
    return PROMPT_Add_Retrieval_Case.format(prompt=init_prompt, retrieval_case_plan=retrieval_case_plan)

def build_determine_continue_prompt(current_step_info, previous_steps_info):
    prompt = f"""
The current step and tool ({current_step_info['tool_name']}) execution is complete. validate the correctness of the parameters or tool execution results? [valid/invalid]
- current step info, including the name of the tool called at each step, parameter information, and the returned result (the type of the result and the specific value)::
{current_step_info}
- previous step info:
{previous_steps_info}
"""
    return prompt

    
if __name__ == '__main__':
    print(build_difficulty_prompt("test_problem"))
    
# Please note:
#   - You should first provide a detailed explanation of your thinking process, using the start tag <think> and end tag </think> to identify the thinking process;
#   - Followed the end tag </think>, provide the specific answer content and pay attention to the line breaks.