import os
import numpy as np
import torch
import json
from openai import OpenAI
import re
import yaml

PROMPT_Standard_Format: str = """
Convert the following PDE problem into the following standard format:
PDE problem:
{PDE_problem}

Standard format:
I would like to solve the following heat equation: specific_PDE_equation
where specific_parameters
The computational geometry domain / time doamin is specific_geometry_domain_and_time_domain
The initial condition specific_initial_condition
The boundary condition specific_boundary_condition

Example:
I would like to solve the following Burgers equation: \\frac{{\\partial u}}{{\\partial t}} + u\\frac{{\\partial u}}{{\\partial x}} = \\nu\\frac{{\\partial^2u}}{{\\partial x^2}}
Where \\nu=0.01
The computational geometry domain is [-1,1], and the time domain is [0,1]
The initial condition \\quad u(x,0) = - \\sin(\\pi x)
The Dirichlet boundary condition u(-1,t) = u(1,t) = 0

Formatting notes:
- If the parameter string is in LaTex format, you need to use LaTex format in the output string, and you should use "\frac" instead of "\\frac" in your output to avoid json.loads() parsing errors.
- If the JSON string contains control characters such as line breaks (\n) and tabs (\t), you need to escape them as \\n or \\t.
"""

PROMPT_Standard_Format: str = """
Convert the following PDE problem into the following standard format:
PDE problem:
{PDE_problem}

Standard format:
I would like to solve the following heat equation: specific_PDE_equation
where specific_parameters
The computational geometry domain / time doamin is specific_geometry_domain_and_time_domain
The initial condition specific_initial_condition
The boundary condition specific_boundary_condition

Example:
I would like to solve the following Burgers equation: \\frac{{\\partial u}}{{\\partial t}} + u\\frac{{\\partial u}}{{\\partial x}} = \\nu\\frac{{\\partial^2u}}{{\\partial x^2}}
Where \\nu=0.01
The computational geometry domain is [-1,1], and the time domain is [0,1]
The initial condition \\quad u(x,0) = - \\sin(\\pi x)
The Dirichlet boundary condition u(-1,t) = u(1,t) = 0

Formatting notes:
- If the parameter string is in LaTex format, you need to use LaTex format in the output string, and you should use "\frac" instead of "\\frac" in your output to avoid json.loads() parsing errors.
- If the JSON string contains control characters such as line breaks (\n) and tabs (\t), you need to escape them as \\n or \\t.
"""

PROMPT_Standard_Format: str = """
Convert the following PDE problem into the following standard format:
PDE problem:
{PDE_problem}

Standard format:
I would like to solve the following heat equation: specific_PDE_equation
where specific_parameters
The computational geometry domain / time doamin is specific_geometry_domain_and_time_domain
The initial condition specific_initial_condition
The boundary condition specific_boundary_condition

Example:
I would like to solve the following Burgers equation: \\frac{{\\partial u}}{{\\partial t}} + u\\frac{{\\partial u}}{{\\partial x}} = \\nu\\frac{{\\partial^2u}}{{\\partial x^2}}
Where \\nu=0.01
The computational geometry domain is [-1,1], and the time domain is [0,1]
The initial condition \\quad u(x,0) = - \\sin(\\pi x)
The Dirichlet boundary condition u(-1,t) = u(1,t) = 0

Formatting notes:
- If the parameter string is in LaTex format, you need to use LaTex format in the output string, and you should use "\frac" instead of "\\frac" in your output to avoid json.loads() parsing errors.
- If the JSON string contains control characters such as line breaks (\n) and tabs (\t), you need to escape them as \\n or \\t.
"""

PROMPT_Standard_Format2: str = """
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

os.environ["CHAT_MODEL"] = "deepseek-chat"
os.environ["DEEPSEEK_API_KEY"] = "sk-e7da98d7c23242509ece68a221e02348"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"

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

with open("exp.json", "r") as f:
    exp_json = json.load(f)
    

    
# for root, dirs, files in os.walk("../database"):
#     # for file in files:
#     #     if file.endswith(".json"):
#     #         print(file)
#     # for file in os.listfiles(root):
#     print(root)
#     print(dirs)
#     for file in os.listdir(root):
#         if file.endswith(".json"):
#             print(file)
PDE_problem_dict = {}
for file in os.listdir("../database"):
    if file.endswith(".json"):
        print(file)
        with open(os.path.join("../database", file), "r", encoding='utf-8') as f:
            data_json = json.load(f)
            
        if data_json.get("problem_description", ""):
            problem_description = data_json["problem_description"]
            prompt = PROMPT_Standard_Format2.format(PDE_problem=problem_description, 
                                                   Standard_format=exp_json["Standard_format"], 
                                                   Example=exp_json["Example"])
            client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], 
                                 base_url=os.environ['DEEPSEEK_BASE_URL'])
            messages = [
                {"role": "user", "content": prompt},]
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0,  # 贪婪策略，固化回答
                top_p=0.8,
                max_tokens=5120,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            content = response.choices[0].message.content
            print(content)
            print("="*50)
            match = re.search(r'<answer>(.*?)</answer>(.*)', content, re.DOTALL)
            if match:
                answer_content = match.group(1).strip().replace("json", "").replace("```", '')
                PDE_problem_dict[file] = answer_content
            else:
                PDE_problem_dict[file] = content
            
with open("./temp_pde.json", "w", encoding='utf-8') as f:
    json.dump(PDE_problem_dict, f, ensure_ascii=False, indent=4)
with open("./temp_pde.yaml", "w", encoding='utf-8') as f:
    yaml.dump(PDE_problem_dict, f, allow_unicode=True)
    


            
            