import random
from tqdm import tqdm
import json
import os
from typing import *
from pathlib import Path
import torch
import tiktoken

from openai import OpenAI
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import transfer_time_to_text as t4
import prompt_engineering as pe
# import age_engineering as ae
from generate_qa_pair import generate_descriptions
from sklearn.model_selection import train_test_split
import copy
import pdb
from zai import ZhipuAiClient
import concurrent.futures
from typing import Optional 
import markdown
import datetime

def delLineBreaks(signal):
    text = str(signal)
    return text.replace('\n', ' ').replace('  ', ' ')

import time

def validate_result(result: str) -> bool:
    """验证结果是否合格"""
    if not result:
        print('no valid results!')
        return False
    try:
        data = json.loads(result)
        print('json has been loaded successfully.')
        if "cells_info" in data.keys():
            print('found cells info')
            pe.check_ind_des(data)
        else:
            print('do not find cells_info')
            return False
        return True
    except:
        print('failed to validate')
        return False
    
def set_all_seeds(seed=42):
    """
    设置所有主要库的随机种子以确保结果可复现
    
    参数:
    seed (int): 随机种子值，默认为42
    """
    # 1. Python内置随机模块
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch (CPU和GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # 5. 设置Python哈希种子（用于字典等数据结构的哈希）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 6. 设置环境变量（某些库会读取）
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 对某些PyTorch操作是必需的


qwen_list = ['qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b-instruct-2507']
# # qwen3
OPENAI_API_KEY = "sk-ab07cd0c228c49aab7897b6e1e8382a5"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_MODEL = qwen_list[0]

# deepseek

# OPENAI_API_KEY = "sk-4eaade62e88a40c09af5b4c139674d26"
# OPENAI_BASE_URL = "https://api.deepseek.com"
# OPENAI_MODEL = 'deepseek-chat'

# OPENAI_MODEL = 'glm-4.5'

if 'glm' in OPENAI_MODEL: 
    # client = ZhipuAiClient(api_key="13ab570616de4823b2777e9b74041126.Xkj0uHXFaGHGH1Fn")
    client = ZhipuAiClient(api_key="0eaf5ebe67a94c7bbd66d2373bc7d53d.xglPgSVXC2EZg0Rr")
    MAX_TOKEN = 40000
else:
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key= OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    MAX_TOKEN = 8192

def process_task(
    system_prompt: str,
    user_content: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> Optional[str]:
    """
    线程任务函数：处理单个API调用，返回JSON结果（或None，若失败）。
    """
    max_retries = 3  # 最大重试次数
    retry_delay_base = 1.0  # 基础延迟秒数（指数退避）
    
    for attempt in range(max_retries + 1):
        try:
            # 构造对话消息
            messages = [
                {"role": "system", "content": f"(IMPORTANT: This is an isolated request.){system_prompt}"},
                {"role": "user", "content": user_content}
            ]
            
            # 调用API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                extra_body={"enable_thinking": False}
            )
            
            result = response.choices[0].message.content.strip()
            
            # 验证结果（关键步骤）
            if validate_result(result):
                return result  # 验证通过则返回
            else:
                print(f"结果验证失败（尝试 {attempt+1}/{max_retries}）: {result[:100]}...")
        
        except Exception as e:
            print(f"API调用失败（尝试 {attempt+1}/{max_retries}）: {type(e).__name__}: {str(e)[:100]}")
        
        # 重试前延迟（指数退避）
        if attempt < max_retries:
            delay = retry_delay_base * (2 ** attempt)  # 指数增长延迟
            time.sleep(delay)  # 添加随机抖动
            
    if attempt == max_retries:
                print(f"⚠️ 任务最终失败: system_prompt='{system_prompt[:100]}...'")
                return None

# OPENAI_MODEL = "glm-4-plus"
# client = ZhipuAiClient(api_key="13ab570616de4823b2777e9b74041126.Xkj0uHXFaGHGH1Fn")

SYSTEM_PROMPT2, template2 = pe.prompt_for_list()


def create_sliding_windows(data_v, data_c, window_size=200):
    """将时间序列数据切分成滑动窗口样本"""
    num_samples = data_v.shape[0] - window_size + 1
    data_v_win = np.stack([data_v[i:i+window_size] for i in range(num_samples)])
    data_c_win = np.stack([data_c[i:i+window_size] for i in range(num_samples)])
    return data_v_win, data_c_win

def split_time_series(data_v, data_c, window_size=200, test_size=0.2, random_state=42):
    # 1. 创建滑动窗口
    data_v_win, data_c_win = create_sliding_windows(data_v, data_c, window_size)
    
    pick_idx = random.sample(range(data_v_win.shape[0]), 5000)
    data_v_win = data_v_win[pick_idx]
    data_c_win = data_c_win[pick_idx]

    # 2. 生成样本索引并随机分割
    indices = np.arange(len(data_v_win))
    train_indices, test_indices  = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True  # 启用随机打乱
    )
    
    # 3. 按索引分割数据
    data_v_win_train = data_v_win[train_indices]
    data_v_win_test = data_v_win[test_indices]
    data_c_win_train = data_c_win[train_indices]
    data_c_win_test = data_c_win[test_indices]
    
    return data_v_win_train, data_v_win_test, data_c_win_train, data_c_win_test

def timeseries_to_list(timeseries, digits: int=6, cp=True):
    if cp:
        result = copy.deepcopy(timeseries)
    else:
        result = timeseries
    if type(result) == np.ndarray:
        result = result.tolist()
    
    if type(result[0]) == float:
        for i in range(len(result)):
            result[i] = round(float(result[i]), 4)
    else:
        for i in range(len(result)):
            result[i] = timeseries_to_list(result[i], digits, cp=False)

    return result

def TimeSeriesText(signal, begin_idx=0, char_state=1):
    signalTextAll = []
    capacity_name = "charge capacity" if char_state == 1 else "discharge capacity"
    for i in range(int(len(signal)/5)):
        signalText = (f"time series of Cell {i+1}:\n voltage: {delLineBreaks(signal[i*5+1])}; temperature: {delLineBreaks(signal[i*5])}; charge capacity: {delLineBreaks(signal[i*5+2])}; discharge capacity: {delLineBreaks(signal[i*5+3])}; current:{delLineBreaks(signal[i*5+4])}.")
        signalTextAll.append(signalText)
    return "\n".join(signalTextAll)

def multivariate_seed_qa(data_v_seen, data_c_seen, data_v_pre, data_c_pre, char_state):
    
    instruction = (f"There are time series collected from LIB cells.")

    time_series_list = [
    data_v_seen[:, i, j]  # 每个元素是形状 (200,) 的时间序列
    for j in range(data_v_seen.shape[2])
    for i in range(data_v_seen.shape[1])]
    time_series_list.append(data_c_seen.flatten())
    
    data_desp_i = {"clu": f"{TimeSeriesText(time_series_list, begin_idx=0, char_state=char_state)}",
                "ind1": f"{TimeSeriesText(time_series_list, begin_idx=0, char_state=char_state)}",
                }

    SYSTEM_PROMPT_list = [SYSTEM_PROMPT2]
    template_list = [template2]
    des_pick = ["ind1"]
    des_json_list = []
    
    tasks = []
    for sys_prompt, template, des_key in zip(SYSTEM_PROMPT_list, template_list, des_pick):
        user_content = instruction + data_desp_i[des_key]  # 构造用户内容（与原逻辑一致）
        sys_prompt_combined = sys_prompt + template
        tasks.append((sys_prompt_combined, user_content))  # 每个任务包含两个参数
    
    # 3. 使用线程池并行执行任务
    des_json_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 提交任务：用lambda传递固定参数（client、model等），executor.map按顺序返回结果
        results = executor.map(
            lambda task: process_task(
                task[0],  # system_prompt（来自任务列表）
                task[1],  # user_content（来自任务列表）
                OPENAI_MODEL,  # 模型名称（固定）
                0.3,   # 温度（固定）
                MAX_TOKEN     # 最大 tokens（固定）
            ),
            tasks  # 任务列表
        )
        
        # 收集结果：过滤掉失败的任务（返回None的）
        des_json_list = [res for res in results if res is not None]

    result = {
        'original list': "\n".join(data_desp_i),
        'time series': time_series_list,
        'descript c2': des_json_list[0],
    }

    return result

class PreprocessedQADataset:
    """仅保留数据预处理逻辑"""
    def __init__(self, jsonl_file):
        self.samples = self._load_data(jsonl_file)
        
    def _load_data(self, jsonl_file):
        current_file_path = Path(__file__)
        jsonl_file = current_file_path.parents[0] / "dataset" / jsonl_file
        samples = []
        with open(jsonl_file, 'r', encoding='latin-1') as f:
            for line in f:
                cleaned_line = line.replace('¡ãC', '°C').replace('¡ã', '°')
                data = json.loads(cleaned_line)
                timeData = data['time series']
                samples.append({
                    'data': timeData,
                })
        return samples

def read_data_from_json(dataset):
    sample_pick = dataset.samples
    data_v_win = []
    data_c_win = []
    for i in range(len(sample_pick)):
        sample_target = sample_pick[i]["data"]
        sample_per = []
        for j in range(6):
            sample_sample_per = np.stack(sample_target[5*j:5*(j+1)])
            sample_per.append(sample_sample_per)
        sample_per = np.stack(sample_per)
        sample_per = np.transpose(sample_per, (2, 1, 0))
        data_v_win.append(sample_per)
        current = np.array(sample_target[-1]).reshape(-1,1)
        data_c_win.append(current)
    return data_v_win, data_c_win

def generate_seed_qa_dataset(data_v_win_train, data_c_win_train):
    generated_cnt = 0
    ts_idx = 0

    with open(OUTPUT_DATASET, 'wt') as f:
        for i in range(TOTAL_CNT):
            print(f"the {i+1} th generation!")
            data_v_i = data_v_win_train[i]
            data_c_i = data_c_win_train[i]
            
            if i >= 20:
                char_state = 0
            else:
                char_state = 1
                
            # data_v_pre = data_v_i[-10:]
            # data_c_pre = data_c_i[-10:]
            # qa_data = generate_descriptions_ano(data_v_pre, data_c_pre)
        
            item = multivariate_seed_qa(data_v_i, data_c_i, 0, 0, char_state)
            
            cur_data = {
                'sample id': i+1,
                'original list': item['original list'],
                'time series': timeseries_to_list(item['time series']),
                'descript c2': item['descript c2'],
            }
            f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')
        ts_idx += 1

if __name__ == '__main__':
    # CONFIG
    set_all_seeds()
    ENCODING_METHOD = 'sp'
    SEQ_LEN = 256  # Set to none for random seq_len
    TOTAL_CNT = 10
    mode = 1
    dataset = PreprocessedQADataset("llama3-description1128.jsonl")
    current_file_path = Path(__file__) 
    dir_path = current_file_path.parents[0] / "dataset"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DATASET = f'{dir_path}\\template_qa_ow{OPENAI_MODEL}_mit_{TOTAL_CNT}_{timestamp[:8]}.jsonl'

    data_v_win_train, data_c_win_train = read_data_from_json(dataset)
    print("info: mit owgeneration, model: ", OPENAI_MODEL)
    generate_seed_qa_dataset(data_v_win_train, data_c_win_train)