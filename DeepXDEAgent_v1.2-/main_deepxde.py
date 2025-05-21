import os
import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
from utils import (
    determine_difficulty,
    process_easy_query, determine_difficulty, process_medium_query, process_difficult_query
)
from options.config_options import ConfigOptions
import logging
from tools.ToolFun_PDE_v2 import *
import yaml
import datetime
import sys

def load_config(file_path):
    # global config
    with open(file_path, 'r', encoding='utf-8') as file:
        print(f"reading file {file_path}...")
        config = yaml.safe_load(file)
    return config

def get_tools_info():
    Tool_Dict = {
        "define_pde": define_pde,
        "define_reference_solution": define_reference_solution,
        "define_domain": define_domain,
        "define_initial_condition": define_initial_condition,
        "define_boundary_condition": define_boundary_condition,
        "create_training_data": create_training_data,
        "create_network": create_network,
        "train_model": train_model,
        "train_model_LBFGS": train_model_LBFGS,
        "visualize_and_save": visualize_and_save
        }
    
    pde_parser_tool = list(Tool_Dict.keys())[:5]
    pde_solver_tool = list(Tool_Dict.keys())[5:]

    # 加载工具函数信息，即tools_PDE.json文件
    tools_info_path = os.path.join(os.path.dirname(__file__), "tools", "tools_PDE_v2.json")
    with open(tools_info_path, "r", encoding="utf-8") as f:
        tools_info = json.load(f)
    f.close()
    
    tools_info_dict = {}
    for tool in tools_info:
        tools_info_dict[tool['name']] = {'description': tool['description'], 'parameters': tool['parameters']}
        
    return {"Tool_Dict": Tool_Dict, "tools_info_dict": tools_info_dict, "pde_parser_tool": pde_parser_tool, "pde_solver_tool": pde_solver_tool}

def main(Debug=False):
    # Parse arguments
    # opt = ConfigOptions()
    # args = opt.initialize()
    
    # config_file_path = os.getenv('CONFIG_FILE_PATH', '')
    config_file_path = "./prompts/config.yaml"
    usr_config_file_path = "./inputs/HeatEquation.yaml"
    config = load_config(config_file_path)
    config.update(load_config(usr_config_file_path))
    config['log_dir'] = os.path.join(os.getcwd(), config.get("output_dir", "logs"))
    now = datetime.datetime.now()
    config['results_dir'] = os.path.join(config['log_dir'], now.strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(config['results_dir']):
        os.makedirs(config['results_dir'])
    os.environ["OUTPUT_DIR"] = config["results_dir"]
    
    if Debug:
        logging.basicConfig(filename=config['log_dir'] + "/" +now.strftime("%Y-%m-%d_%H-%M-%S") + ".log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s\n', datefmt='%H:%M:%S', encoding='utf-8')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info("Start processing...")
        logging.info("#"*10 + "1 基本配置" + "#"*10)
        logging.info(f"config_file_path: {config_file_path}")
        logging.info(f"config: \n{config}")

    # model, client = setup_model(args.model)
    # test_qa, examplers = load_data(args.dataset)
    # Set environment variables from config
    chat_model_info = config.get('model_info', {})
    if not chat_model_info:
        raise ValueError("No model_info found in config.")
    if chat_model_info['CHAT_MODEL'].lower().startswith("deepseek"):
        os.environ["CHAT_MODEL"] = chat_model_info['CHAT_MODEL']
        os.environ["DEEPSEEK_API_KEY"] = chat_model_info.get("DEEPSEEK_API_KEY", "")
        os.environ["DEEPSEEK_BASE_URL"] = chat_model_info.get("DEEPSEEK_BASE_URL", "")
    elif chat_model_info['CHAT_MODEL'].lower().startswith("gpt"):
        os.environ["CHAT_MODEL"] = chat_model_info['CHAT_MODEL']
        os.environ["OPENAI_API_KEY"] = chat_model_info.get("OPENAI_API_KEY", "")
        os.environ["OPENAI_PROXY"] = chat_model_info.get("OPENAI_PROXY", "")
        os.environ["OPENAI_BASE_URL"] = chat_model_info.get("OPENAI_BASE_URL", "")
    else:
        print('the chat model is not identified, only the model start with deepseek and gpt can be identified.')
        sys.exit()

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)
    
    # 定义工具与说明
    # Tool_Dict, tools_info_dict = get_tools_info()
    # config['Tool_Dict'], config['tools_info_dict'] = Tool_Dict, tools_info_dict
    config.update(get_tools_info())
    Tool_Dict, tools_info_dict = config['Tool_Dict'], config['tools_info_dict']
    
    user_input = config['user_input']
    difficulty = config.get('difficulty_level', 'adaptive')
    # ['easy', 'medium', ‘difficult']
    difficulty = determine_difficulty(user_input, difficulty, Debug)
    if Debug:
        logging.info(f"user_input: \n{user_input}")
        logging.info(f"difficulty: {difficulty}")
        logging.info(f"Tool_Dict: \n{Tool_Dict}")
        logging.info(f"tools_info_dict: \n{tools_info_dict}")
        
    # TODO: 测试
    difficulty = 'difficult'
    if difficulty == 'easy':
        final_decision, planning_message, implement_message = process_easy_query(user_input, config['thinking_prompt'], config, Debug)
        # final_decision = process_basic_query(user_input, examplers, args.model, args)
        
    elif difficulty =='medium':
        final_decision, planning_message, implement_message  = process_medium_query(user_input, config, Debug)
    elif difficulty == 'difficult':
        final_decision, planning_message, implement_message = process_difficult_query(user_input, config, Debug)
    if Debug:
        logging.info("#"*5 + "Final Output" + "#"*5)
        logging.info(f"final_decision: \n{final_decision}")
        logging.info("\n" + "#"*5 + "Planning Message" + "#"*5)
        for idx, item in enumerate(planning_message):
            logging.info(f"role: {item['role']} | \n{item['content']}")
        logging.info("\n" + "#"*5 + "Implement Message" + "#"*5)
        for idx, item in enumerate(implement_message):
            logging.info(f"role: {item['role']} | \n{item['content']}")
    return final_decision


    
        
if __name__ == '__main__':
    main(Debug=True)
    
    