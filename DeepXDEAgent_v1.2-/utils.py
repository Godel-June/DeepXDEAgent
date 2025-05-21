import os
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node
import google.generativeai as genai
from openai import OpenAI
from pptree import *
from prompts.prompts import *
import logging
import re
from matplotlib import pyplot as plt

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatOpenAI
import sys
from langchain.chains import RetrievalQA

class Agent:
    def __init__(self, instruction, role, model_info=None, examplers=None, img_path=None, is_recorded=False):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info if model_info is not None else os.getenv('CHAT_MODEL', 'gpt-4o-mini')
        self.img_path = img_path
        self.is_recorded = is_recorded

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'deepseek-chat']:
            # self.client = OpenAI(api_key=model_info['deepseek_api_key'], 
            #                      base_url=model_info['deepseek_base_url']) if self.model_info=='deepseek-chat' else OpenAI(api_key=model_info['openai_api_key'])
            self.client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], 
                                 base_url=os.environ['DEEPSEEK_BASE_URL']) if self.model_info=='deepseek-chat' else OpenAI(api_key=os.environ['OPENAI_API_KEY'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
        if self.is_recorded:
            self.record_path = os.path.join(os.getenv('OUTPUT_DIR'), f'{self.role}_record.txt')
            with open(self.record_path, 'w', encoding='utf-8') as f:
                for item in self.messages:
                    f.write("#"*5 + item['role'] + ':'+ '\n' + item['content'] + '\n\n')

    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'deepseek-chat']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            elif self.model_info == 'deepseek-chat':
                model_name = "deepseek-chat"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages,
                temperature=0,  # 贪婪策略，固化回答
                top_p=0.8,
                max_tokens=5120,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            
            if self.is_recorded:
                with open(self.record_path, 'a', encoding='utf-8') as f:
                    f.write("user: " + '\n' + message + '\n\n')
                    f.write("assistant: " + '\n' + response.choices[0].message.content + '\n\n')
            
            think_content, answer_content = self.response_split(response.choices[0].message.content)
            self.messages.append({"role": "assistant", "content": answer_content})
            return think_content, answer_content
        
    def response_split(self, content: str) -> tuple:
        match = re.search(r'<think>(.*?)</think>(.*)', content, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip().replace("json", "").replace("```", '')
            return think_content, answer_content
        else:
            return "未找到有效的<think>标签", content

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.0]
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4o-mini'
                response = self.client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                
                responses[temperature] = response.choices[0].message.content
                
            return responses
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses


def determine_difficulty(question, difficulty, Debug=False):
    if difficulty != 'adaptive':
        return difficulty
    
    # difficulty_prompt = f"""Now, given the differential equation as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of solving the differential equation with DeepXDE using among below options:\n1) easy: a single agent using DeepXDE can output an answer.\n2) medium: a single agent with RAG using DeepXDE can output an answer.\n3) ‘difficult: multiple agent with RAG using DeepXDE can work together to make final answer."""
    difficulty_prompt = build_difficulty_prompt(question)
    
    math_agent = Agent(instruction='You are a math expert who conducts initial assessment and your job is to decide the difficulty/complexity of solving differential equation with DeepXDE.', role='math expert', moded_info='deepseek')
    # math_agent.chat('You are a math expert who conducts initial assessment and your job is to decide the difficulty/complexity of the math query.')
    response = math_agent.chat(difficulty_prompt)
    
    if Debug:
        logging.info(f"Difficulty prompt: {difficulty_prompt}")
        logging.info(f"Difficulty response: \n{response}")

    # 1) Easy, 2) Medium, or 3) Difficult
    if 'easy' in response.lower() or '1)' in response.lower():
        return 'easy'
    elif 'medium' in response.lower() or '2)' in response.lower():
        return 'medium'
    elif 'difficult' in response.lower() or '3)' in response.lower():
        return 'difficult'
    
def parse_tool_params(tool_params: dict, previous_steps_info: list) -> dict:
    """
    解析工具参数，返回一个 kwargs 字典，方便后续直接通过 func(**kwargs) 调用工具。

    如果某个参数的值为字符串且符合 "Step_n_m" 格式，则用 previous_steps_info 中对应步骤的返回值替换，
    否则直接使用原始值（JSON 解析时已经保证了数值、bool等类型正确）。

    :param tool_params: 工具所需参数的字典，格式如：
        {
            "param_name": {
                "type": "类型描述",
                "value": "Step_n_m" 或直接值
            },
            ...
        }
    :param previous_steps_info: 一个列表，其中每个元素都是一个字典，包含工具名称、参数信息和返回结果，如：
        {
            "step": 步数（从 1 开始）,
            "tool_name": "工具A",
            "params": "参数信息"
            "return1": {"type": "...", "value": "value1"},
            "return2": {"type": "...", "value": "value2"},
            ...
        }
    :return: kwargs 字典，其中每个键对应一个参数名，值为解析后的参数值
    """
    kwargs = {}
    step_pattern = re.compile(r"Step_(\d+)_(\d+)")  # 匹配 "Step_n_m" 格式

    for param_name, param_info in tool_params.items():
        param_value = param_info["value"]

        # 仅当参数值为字符串时检查是否为 "Step_n_m" 格式
        if isinstance(param_value, str):
            match = step_pattern.fullmatch(param_value)
            if match:
                step_index = int(match.group(1)) - 1  # 转换为列表索引（从 0 开始）
                return_key = f"return{int(match.group(2))}"  # 构造返回值键，例如 "return1"
                try:
                    param_value = previous_steps_info[step_index][return_key]["value"]
                except (IndexError, KeyError):
                    raise ValueError(f"Invalid reference: {param_value} not found in previous_steps_info")

        kwargs[param_name] = param_value

    return kwargs

def execute_tool(Tool_Dict: dict, tool_name: str, kwargs: dict) -> dict:
    """
    执行工具，返回一个字典，包含工具的返回结果。
    :param tool_name: 待调用的工具名称
    :param kwargs: 待调用工具的参数字典
    :return:
    """
    tool_func = Tool_Dict[tool_name]
    # 如果没有找到工具，输出错误
    if tool_func is None:
        raise ValueError(f"Tool {tool_name} Not Found in Toolset")

    return_value = tool_func(**kwargs)

    # 根据返回值是否为元组，生成对应的结果字典
    if isinstance(return_value, tuple):
        result_dict = {}
        for i, val in enumerate(return_value):
            if isinstance(val, list):
                result_dict[f"return{i + 1}"] = {
                    "type": [type(item).__name__ for item in val],
                    "value": val
                }
            else:
                result_dict[f"return{i + 1}"] = {"type": type(val).__name__, "value": val}
    elif isinstance(return_value, list):
        result_dict = {"return1": {"type": [type(item).__name__ for item in return_value], "value": return_value}}
    else:
        result_dict = {"return1": {"type": type(return_value).__name__, "value": return_value}}

    return result_dict
    
def process_easy_query(question, instruction, config, Debug=False):
    math_agent = Agent(instruction=instruction.strip(), role='math expert', is_recorded=True)
    task_split_prompt = build_query_split_prompt(question, config['tools_info_dict'])
    if Debug:
        logging.info("\n" + "#"*10 + " 1 任务规划 " + "#"*10)
        logging.info("planning agent 已创建")
        logging.info(f"Task split prompt: \n{task_split_prompt}")
    task_split_think, task_split_answer = math_agent.chat(task_split_prompt)
    # 解析任务分解结果，转换为列表
    task_split_info = json.loads(task_split_answer)
    # TODO: logging info 输出任务拆分信息
    if Debug:
        logging.info(f"\n任务规划完成：")
        logging.info(f"🤔 任务拆分\n🧠 任务拆分思考过程：\n{task_split_think}")
        # logging.info(f"Task split think: \n{task_split_think}")
        logging.info(f"Task split answer: \n{task_split_answer}")
        logging.info(f"任务拆分结果：\n{task_split_info}")
        
        logging.info("\n" + "#"*10 + " 2 工具调用 " + "#"*10)
        logging.info("工具 agent 创建：")
    
    implement_agent = Agent(instruction=config['implement_prompt'].strip().format(user_input=config['user_input']), role='Implement Agent', is_recorded=True)
    # 顺序处理每个任务
    previous_steps_info = []  # 工具执行结果列表，用于参数传递
    for idx, task_info in enumerate(task_split_info):
        # 得到当前待调用工具
        tool_name = task_info["tool_name"]
        # 得到当前任务描述
        task_desc = task_info["reasoning"]
        # 得到当前工具详情
        tool_info = [item for item in config['tools_info_dict'].items() if item[0] == tool_name]
        tool_info = tool_info[0] if tool_info else None
        
        # 如果没有找到工具信息，输出错误信息
        if not tool_info:
            # st.error("⚠️ 未在工具库中找到该工具，请检查工具名称是否正确。")
            print(f"\n⚠️ 未在工具库中找到该工具，请检查工具名称是否正确。\n")
            logging.error(f"⚠️ 未在工具库中找到该工具，请检查工具名称是否正确。")
            raise ValueError(f"Invalid tool name: {tool_name}")
        # print(f"\n已在工具库中找到工具：{tool_name}\n")
        logging.info("#"*5 + f" 已在工具库中找到工具：{tool_name}")
        
        # 确定工具参数
        tool_params_prompt = determine_tool_params_prompt(tool_info, previous_steps_info)
        tool_params_think, tool_params_answer = implement_agent.chat(tool_params_prompt)
        # 工具解析
        tool_params = json.loads(tool_params_answer)
        tool_params_kwargs = parse_tool_params(tool_params, previous_steps_info)
        if Debug:
            logging.info(f"Tool info: {tool_info}")
            logging.info(f"\n🛠️ 工具选择：{tool_name}")
            logging.info(f"\tTool params prompt: \n{tool_params_prompt}")
            # logging.info(f"Tool params answer: \n{tool_params_answer}")
            # logging.info(f"Tool params: {tool_params}")
            # logging.info(f"Tool params kwargs: {tool_params_kwargs}")
            
            logging.info(f"\t🧠 确定工具参数：\n{tool_params_think}")
            logging.info(f"\t参数：\n{tool_params_answer}")
            logging.info(f"\t输出参数 json: {tool_params}")
            logging.info(f"\t参数解析结果：{tool_params_kwargs}")
            
            logging.info(f"\n工具{tool_name}参数解析完成")
            # logging.info(f"工具{tool_name}参数：{tool_params_kwargs}")
            
        # 执行工具
        tool_result = execute_tool(config['Tool_Dict'], tool_name, tool_params_kwargs)
        
        # 记录工具执行结果
        previous_steps_info.append({
            "step": idx+1,
            "tool_name": tool_name,
            "params": tool_params_kwargs,
            **tool_result
        })
        
        if Debug:
            logging.info(f"\n工具 {tool_name} 执行完成，返回结果：\n{tool_result}")
            logging.info(f"\n✅ 执行结果：")
            figure_flag = False
            for id, (key, item) in enumerate(tool_result.items()):
                if isinstance(item["value"], plt.Figure):
                    figure_flag = True
                    # plt.plot(item["value"])  # 显示 Matplotlib 图像
                    # plt.show()
                    plt.figure(item["value"].figure)  # 指定要显示的图形
                    # plt.show()  # 显示 Matplotlib 图像
                    logging.info(f"\t工具执行结果{id + 1}：图像已展示")
                    plt.savefig(os.path.join(os.getenv('OUTPUT_DIR'), f"{tool_name}_{id+1}.png"))
                else:
                    print(f"\t工具执行结果{id + 1}：{item['value']}")  # 其他类型直接输出
                    logging.info(f"\t工具执行结果{id + 1}：\n{item['value']}")
            if figure_flag:
                plt.show()  # 显示 Matplotlib 图像
            # logging.info(f"\n{tool_name}执行完成")
            logging.info(f"当前记录的所有工具结果：\n{previous_steps_info}")

    logging.info("\n所有步骤执行完成")
    return previous_steps_info, math_agent.messages, implement_agent.messages


def process_medium_query(question, config, Debug=False):
    # retrieval module
    pde_problem_directory = "./database/pde_problem_faiss"
    pde_case_directory = "./database/pde_case_faiss"
    # local_model_path = "tbs17/MathBERT"
    local_model_path = r"C:\Users\jermain\.cache\huggingface\hub\models--tbs17--MathBERT\snapshots\e26235ccf2b14614ef278b19caa44fdb5dcf050f"
    pde_problem_vectordb = FAISS.load_local(pde_problem_directory, HuggingFaceEmbeddings(model_name=local_model_path),allow_dangerous_deserialization=True)
    pde_case_vectordb = FAISS.load_local(pde_case_directory, HuggingFaceEmbeddings(model_name=local_model_path),allow_dangerous_deserialization=True)
    
    chat_model_name = config['model_info'].get('CHAT_MODEL', '')
    temperature = 0.0
    if chat_model_name.startswith("deepseek"):
        chat_model = ChatDeepSeek(model=chat_model_name, temperature=temperature)
    elif chat_model_name.lower().startswith("gpt"):
        chat_model = ChatOpenAI(model=chat_model_name, temperature=temperature)
    else:
        print('the chat model is not identified, only the model start with deepseek and gpt can be identified.')
        sys.exit()
    
    # k = config['searchdocs']
    k = 2
    retriever = pde_problem_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa_interface = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    prompt_translate = build_standard_format_prompt(question)
    rsp = qa_interface(prompt_translate)
    user_input = rsp['result']
    # 将其提取出来
    match = re.search(r'<answer>(.*?)</answer>(.*)', user_input, re.DOTALL)
    if match:
        user_input = match.group(1).strip().replace("json", "").replace("```", '')
    else:
        raise ValueError("No answer found in the response.")
    if Debug:
        logging.info("#"*5 + f" Translate Format")
        logging.info(f"question: {question}")
        logging.info(f"prompt_translate: {prompt_translate}")
        logging.info(f"rsp answer: {rsp}")
        logging.info(f"Formated user_input: {user_input}")
    
    logging.info(f"format result: {user_input}")
    prompt_retrieval = build_retrieval_prompt(user_input)
    rsp = qa_interface(prompt_retrieval)
    retrieval_doc = rsp['source_documents'][0]  # Document 对象
    logging.info(f"retrieval result: {retrieval_doc}")
    
    retrieval_pde_case = pde_case_vectordb.similarity_search(query="", k=1,filter={"uid": retrieval_doc.metadata['uid']})[0].page_content
    retrieval_pde_case = json.loads(retrieval_pde_case)
    if Debug:
        # os.path.join(os.getenv('OUTPUT_DIR'), "retrieval_pde_case.json")
        with open(os.path.join(os.getenv('OUTPUT_DIR'), "retrieval_pde_case.json"), "w", encoding="utf-8") as f:
            json.dump(retrieval_pde_case, f, ensure_ascii=False, indent=4)
        logging.info(f"retrieval_pde_case: {retrieval_pde_case}")
    
    # 规划任务
    planning_agent = Agent(instruction=config['thinking_prompt'].strip(), role='planning agent', is_recorded=True)
    task_split_prompt = add_retrieval_case_to_prompt(question, retrieval_pde_case, config['tools_info_dict'])
    
    # 输出规划结果
    if Debug:
        logging.info("\n" + "#"*10 + " 1 任务规划 " + "#"*10)
        logging.info("planning agent 已创建")
        logging.info(f"Task split prompt: \n{task_split_prompt}")
    task_split_think, task_split_answer = planning_agent.chat(task_split_prompt)
    # 解析任务分解结果，转换为列表
    task_split_info = json.loads(task_split_answer)
    # TODO: logging info 输出任务拆分信息
    if Debug:
        logging.info(f"\n任务规划完成：")
        logging.info(f"🤔 任务拆分\n🧠 任务拆分思考过程：\n{task_split_think}")
        # logging.info(f"Task split think: \n{task_split_think}")
        logging.info(f"Task split answer: \n{task_split_answer}")
        logging.info(f"任务拆分结果：\n{task_split_info}")
        
        logging.info("\n" + "#"*10 + " 2 工具调用 " + "#"*10)
        logging.info("工具 agent 创建：")
        
    # 开始工具使用
    implement_agent = Agent(instruction=config['implement_prompt'].strip().format(user_input=config['user_input']), role='Implement Agent', is_recorded=True)
    # 顺序处理每个任务
    previous_steps_info = []  # 工具执行结果列表，用于参数传递
    for idx, task_info in enumerate(task_split_info):
        # 得到当前待调用工具
        tool_name = task_info["tool_name"]
        # 得到当前任务描述
        task_desc = task_info["reasoning"]
        # 得到当前工具详情
        tool_info = [item for item in config['tools_info_dict'].items() if item[0] == tool_name]
        tool_info = tool_info[0] if tool_info else None
        
        # 如果没有找到工具信息，输出错误信息
        if not tool_info:
            # st.error("⚠️ 未在工具库中找到该工具，请检查工具名称是否正确。")
            print(f"\n⚠️ 未在工具库中找到该工具，请检查工具名称是否正确。\n")
            logging.error(f"⚠️ 未在工具库中找到该工具，请检查工具名称是否正确。")
            raise ValueError(f"Invalid tool name: {tool_name}")
        # print(f"\n已在工具库中找到工具：{tool_name}\n")
        logging.info("#"*5 + f" 已在工具库中找到工具：{tool_name}")
        
        # 确定工具参数
        tool_params_prompt = determine_tool_params_prompt(tool_info, previous_steps_info)
        tool_params_think, tool_params_answer = implement_agent.chat(tool_params_prompt)
        # 工具解析
        tool_params = json.loads(tool_params_answer)
        tool_params_kwargs = parse_tool_params(tool_params, previous_steps_info)
        if Debug:
            logging.info(f"Tool info: {tool_info}")
            logging.info(f"\n🛠️ 工具选择：{tool_name}")
            logging.info(f"\tTool params prompt: \n{tool_params_prompt}")
            # logging.info(f"Tool params answer: \n{tool_params_answer}")
            # logging.info(f"Tool params: {tool_params}")
            # logging.info(f"Tool params kwargs: {tool_params_kwargs}")
            
            logging.info(f"\t🧠 确定工具参数：\n{tool_params_think}")
            logging.info(f"\t参数：\n{tool_params_answer}")
            logging.info(f"\t输出参数 json: {tool_params}")
            logging.info(f"\t参数解析结果：{tool_params_kwargs}")
            
            logging.info(f"\n工具{tool_name}参数解析完成")
            # logging.info(f"工具{tool_name}参数：{tool_params_kwargs}")
            
        # 执行工具
        tool_result = execute_tool(config['Tool_Dict'], tool_name, tool_params_kwargs)
        
        # 记录工具执行结果
        previous_steps_info.append({
            "step": idx+1,
            "tool_name": tool_name,
            "params": tool_params_kwargs,
            **tool_result
        })
        
        if Debug:
            logging.info(f"\n工具 {tool_name} 执行完成，返回结果：\n{tool_result}")
            logging.info(f"\n✅ 执行结果：")
            figure_flag = False
            for id, (key, item) in enumerate(tool_result.items()):
                if isinstance(item["value"], plt.Figure):
                    figure_flag = True
                    # plt.plot(item["value"])  # 显示 Matplotlib 图像
                    # plt.show()
                    plt.figure(item["value"].figure)  # 指定要显示的图形
                    # plt.show()  # 显示 Matplotlib 图像
                    logging.info(f"\t工具执行结果{id + 1}：图像已展示")
                    plt.savefig(os.path.join(os.getenv('OUTPUT_DIR'), f"{tool_name}_{id+1}.png"))
                else:
                    print(f"\t工具执行结果{id + 1}：{item['value']}")  # 其他类型直接输出
                    logging.info(f"\t工具执行结果{id + 1}：\n{item['value']}")
            if figure_flag:
                plt.show()  # 显示 Matplotlib 图像
            # logging.info(f"\n{tool_name}执行完成")
            logging.info(f"当前记录的所有工具结果：\n{previous_steps_info}")

    logging.info("\n所有步骤执行完成")
    return previous_steps_info, planning_agent.messages, implement_agent.messages


def process_difficult_query(question, config, Debug=False):
    # TODO: 1 retrieval module
    pde_problem_directory = "./database/pde_problem_faiss"
    pde_case_directory = "./database/pde_case_faiss"
    # local_model_path = "tbs17/MathBERT"
    local_model_path = r"C:\Users\jermain\.cache\huggingface\hub\models--tbs17--MathBERT\snapshots\e26235ccf2b14614ef278b19caa44fdb5dcf050f"
    pde_problem_vectordb = FAISS.load_local(pde_problem_directory, HuggingFaceEmbeddings(model_name=local_model_path),allow_dangerous_deserialization=True)
    pde_case_vectordb = FAISS.load_local(pde_case_directory, HuggingFaceEmbeddings(model_name=local_model_path),allow_dangerous_deserialization=True)
    
    chat_model_name = config['model_info'].get('CHAT_MODEL', '')
    temperature = 0.0
    if chat_model_name.startswith("deepseek"):
        chat_model = ChatDeepSeek(model=chat_model_name, temperature=temperature)
    elif chat_model_name.lower().startswith("gpt"):
        chat_model = ChatOpenAI(model=chat_model_name, temperature=temperature)
    else:
        print('the chat model is not identified, only the model start with deepseek and gpt can be identified.')
        sys.exit()
    
    # k = config['searchdocs']
    k = 2
    retriever = pde_problem_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa_interface = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    prompt_translate = build_standard_format_prompt(question)
    rsp = qa_interface(prompt_translate)
    user_input = rsp['result']
    # 将其提取出来
    match = re.search(r'<answer>(.*?)</answer>(.*)', user_input, re.DOTALL)
    if match:
        user_input = match.group(1).strip().replace("json", "").replace("```", '')
    else:
        raise ValueError("No answer found in the response.")
    if Debug:
        logging.info("#"*5 + f" Translate Format")
        logging.info(f"question: {question}")
        logging.info(f"prompt_translate: {prompt_translate}")
        logging.info(f"rsp answer: {rsp}")
        logging.info(f"Formated user_input: {user_input}")
    
    logging.info(f"format result: {user_input}")
    prompt_retrieval = build_retrieval_prompt(user_input)
    rsp = qa_interface(prompt_retrieval)
    retrieval_doc = rsp['source_documents'][0]  # Document 对象
    logging.info(f"retrieval result: {retrieval_doc}")
    
    retrieval_pde_case = pde_case_vectordb.similarity_search(query="", k=1,filter={"uid": retrieval_doc.metadata['uid']})[0].page_content
    retrieval_pde_case = json.loads(retrieval_pde_case)
    if Debug:
        # os.path.join(os.getenv('OUTPUT_DIR'), "retrieval_pde_case.json")
        with open(os.path.join(os.getenv('OUTPUT_DIR'), "retrieval_pde_case.json"), "w", encoding="utf-8") as f:
            json.dump(retrieval_pde_case, f, ensure_ascii=False, indent=4)
        logging.info(f"retrieval_pde_case: {retrieval_pde_case}")
    
    # TODO: 2 规划任务
    planning_agent = Agent(instruction=config['thinking_prompt'].strip(), role='planning agent', is_recorded=True)
    task_split_prompt = add_retrieval_case_to_prompt(question, retrieval_pde_case, config['tools_info_dict'])
    
    # 输出规划结果
    if Debug:
        logging.info("\n" + "#"*10 + " 1 任务规划 " + "#"*10)
        logging.info("planning agent 已创建")
        logging.info(f"Task split prompt: \n{task_split_prompt}")
    task_split_think, task_split_answer = planning_agent.chat(task_split_prompt)
    # 解析任务分解结果，转换为列表
    task_split_info = json.loads(task_split_answer)
    # : logging info 输出任务拆分信息
    if Debug:
        logging.info(f"\n任务规划完成：")
        logging.info(f"🤔 任务拆分\n🧠 任务拆分思考过程：\n{task_split_think}")
        # logging.info(f"Task split think: \n{task_split_think}")
        logging.info(f"Task split answer: \n{task_split_answer}")
        logging.info(f"任务拆分结果：\n{task_split_info}")
        
        logging.info("\n" + "#"*10 + " 2 工具调用 " + "#"*10)
        logging.info("工具 agent 创建：")
    
    # TODO: 3 工具使用
    # 分3个模块：a. 总策略，b. PDE 对象提取，c. PINN求解
    # 开始工具使用
    if retrieval_pde_case:
        rag_case_prompt = config['rag_case_prompt'].strip().format(
            RAG_case_problem=retrieval_pde_case["problem_description"], 
            RAG_case_solution=''.join('\n' + json.dumps(retrieval_pde_case["expected_tool_chain"], ensure_ascii=False, indent=4) + '\n'))
    else:
        rag_case_prompt = ''
    # prompt
    pde_orchestrator_prompt = config['pde_orchestrator_prompt'].strip().format(user_input=user_input, task_split_procedure=task_split_answer, RAG_case_prompt=rag_case_prompt)
    pde_parser_prompt = config['pde_parser_prompt'].strip().format(user_input=user_input)
    pde_solver_prompt = config['pde_solver_prompt'].strip().format(user_input=user_input)
    
    pde_orchestrator_agent = Agent(instruction=pde_orchestrator_prompt, role='PDE Orchestrator', is_recorded=True)
    pde_parser_agent = Agent(instruction=pde_parser_prompt, role='PDE Parser', is_recorded=True)
    pde_solver_agent = Agent(instruction=pde_solver_prompt, role='PDE Solver', is_recorded=True)
    
    # 调用工具
    # 顺序处理每个任务
    # {step: x, agent: xxx, tool_name: xxx, "params":xxx, **tool_result}
    previous_steps_info = []  # 工具执行结果列表，用于参数传递
    pde_parser_tool_list = config['pde_parser_tool']
    for idx, task_info in enumerate(task_split_info):
        # 得到当前待调用工具
        tool_name = task_info["tool_name"]
        # 得到当前任务描述
        task_desc = task_info["reasoning"]
        # 得到当前工具详情
        tool_info = [item for item in config['tools_info_dict'].items() if item[0] == tool_name]
        tool_info = tool_info[0] if tool_info else None
        
        # 如果没有找到工具信息，输出错误信息
        if not tool_info:
            # st.error("⚠️ 未在工具库中找到该工具，请检查工具名称是否正确。")
            print(f"\n⚠️ 未在工具库中找到该工具，请检查工具名称是否正确。\n")
            logging.error(f"⚠️ 未在工具库中找到该工具，请检查工具名称是否正确。")
            raise ValueError(f"Invalid tool name: {tool_name}")
        # print(f"\n已在工具库中找到工具：{tool_name}\n")
        logging.info("#"*5 + f" 已在工具库中找到工具：{tool_name}")
        
        # 确定工具参数
        tool_params_prompt = determine_tool_params_prompt(tool_info, previous_steps_info)
        
        if tool_name in pde_parser_tool_list:
            implement_agent = pde_parser_agent
        else:
            implement_agent = pde_solver_agent
            
        tool_params_think, tool_params_answer = implement_agent.chat(tool_params_prompt)
        # 工具解析
        tool_params = json.loads(tool_params_answer)
        tool_params_kwargs = parse_tool_params(tool_params, previous_steps_info)
        if Debug:
            logging.info(f"Tool info: {tool_info}")
            logging.info(f"\n🛠️ 工具选择：{tool_name}")
            logging.info(f"\tTool params prompt: \n{tool_params_prompt}")
            # logging.info(f"Tool params answer: \n{tool_params_answer}")
            # logging.info(f"Tool params: {tool_params}")
            # logging.info(f"Tool params kwargs: {tool_params_kwargs}")
            
            logging.info(f"\t🧠 确定工具参数：\n{tool_params_think}")
            logging.info(f"\t参数：\n{tool_params_answer}")
            logging.info(f"\t输出参数 json: {tool_params}")
            logging.info(f"\t参数解析结果：{tool_params_kwargs}")
            
            logging.info(f"\n工具{tool_name}参数解析完成")
            # logging.info(f"工具{tool_name}参数：{tool_params_kwargs}")
            
        # 执行工具
        tool_result = execute_tool(config['Tool_Dict'], tool_name, tool_params_kwargs)
        
        current_step_info = {
            "step": idx+1,
            "agent": implement_agent.role,
            "tool_name": tool_name,
            "params": tool_params_kwargs,
            **tool_result
        }
        
        # TODO: 4 确定有效性
        # 确定参数是否合适
        determine_continue_prompt = build_determine_continue_prompt(current_step_info, previous_steps_info)
        determine_continue_think, determine_continue_answer = pde_orchestrator_agent.chat(determine_continue_prompt)
        # 解析结果
        determine_continue_answer = determine_continue_answer.strip().replace("json", "").replace("```", '')
        determine_continue_result = json.loads(determine_continue_answer)
        validation = False if determine_continue_result['validation'].lower() == "invalid" else True
        if Debug:
            logging.info(f"\n🧠 确定是否继续：\n{determine_continue_result['validation']}")
        
        turn_idx = 1
        while not validation and turn_idx < 3:
            # 重新输入参数
            tool_params_prompt = redetermine_tool_params_prompt(tool_info, previous_steps_info, determine_continue_result)
            tool_params_think, tool_params_answer = implement_agent.chat(tool_params_prompt)
            # 工具解析
            tool_params = json.loads(tool_params_answer)
            tool_params_kwargs = parse_tool_params(tool_params, previous_steps_info)
            if Debug:
                logging.info("#" * 5 + f" 重新确定工具参数和执行结果")
            
            # 执行工具
            tool_result = execute_tool(config['Tool_Dict'], tool_name, tool_params_kwargs)
            
            current_step_info = {
                "step": idx+1,
                "agent": implement_agent.role,
                "tool_name": tool_name,
                "params": tool_params_kwargs,
                **tool_result
            }
            # 确定参数是否合适
            determine_continue_prompt = build_determine_continue_prompt(current_step_info, previous_steps_info)
            determine_continue_think, determine_continue_answer = pde_orchestrator_agent.chat(determine_continue_prompt)
            # 解析结果
            determine_continue_answer = determine_continue_answer.strip().replace("json", "").replace("```", '')
            determine_continue_result = json.loads(determine_continue_answer)
            validation = False if determine_continue_result['validation'].lower() == "invalid" else True
            if Debug:
                logging.info(f"\n🧠 确定是否继续：\n{determine_continue_result['validation']}")
                
        # 记录工具执行结果
        # previous_steps_info.append({
        #     "step": idx+1,
        #     "tool_name": tool_name,
        #     "params": tool_params_kwargs,
        #     **tool_result
        # })
        previous_steps_info.append(current_step_info)
        
        if Debug:
            logging.info(f"\n工具 {tool_name} 执行完成，返回结果：\n{tool_result}")
            logging.info(f"\n✅ 执行结果：")
            figure_flag = False
            for id, (key, item) in enumerate(tool_result.items()):
                if isinstance(item["value"], plt.Figure):
                    figure_flag = True
                    # plt.plot(item["value"])  # 显示 Matplotlib 图像
                    # plt.show()
                    plt.figure(item["value"].figure)  # 指定要显示的图形
                    # plt.show()  # 显示 Matplotlib 图像
                    logging.info(f"\t工具执行结果{id + 1}：图像已展示")
                    plt.savefig(os.path.join(os.getenv('OUTPUT_DIR'), f"{tool_name}_{id+1}.png"))
                else:
                    print(f"\t工具执行结果{id + 1}：{item['value']}")  # 其他类型直接输出
                    logging.info(f"\t工具执行结果{id + 1}：\n{item['value']}")
            if figure_flag:
                plt.show()  # 显示 Matplotlib 图像
            # logging.info(f"\n{tool_name}执行完成")
            logging.info(f"当前记录的所有工具结果：\n{previous_steps_info}")

    logging.info("\n所有步骤执行完成")
    return previous_steps_info, planning_agent.messages, implement_agent.messages
    
