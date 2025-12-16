# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 13:50:27 2025

@author: 14525
"""

import os
from openai import OpenAI
import base64
from io import BytesIO
from pathlib import Path
import matplotlib
# 关键修复：设置非交互式后端（Agg），禁用 GUI 窗口，支持线程安全的图像保存
matplotlib.use('Agg')  # 必须在导入 matplotlib.pyplot 之前执行！
import matplotlib.pyplot as plt
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from zai import ZhipuAiClient
import re
import numpy as np

OPENAI_API_KEY = "sk-e76205eae508447dbe165ad68b158858"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


OPENAI_MODEL = "qwen3-vl-plus"
# OPENAI_MODEL = "glm-4.5v"
if 'glm' in OPENAI_MODEL: 
    client = ZhipuAiClient(api_key="13ab570616de4823b2777e9b74041126.Xkj0uHXFaGHGH1Fn")
else:
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key= OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

def TimeSeriesImage(signal, cell_num=16):
    """
    生成时间序列图像并返回 base64 编码列表（线程安全版本）
    
    :param signal: 输入信号数据（假设格式与原代码兼容）
    :param cell_num: 电池单元数量
    :return: 包含所有图像 base64 字符串的列表
    """
    cols_name = ["temperture", "voltage", "SOC"]
    unit_name = ["°C", "V","%"]
    base64_images = []
    
    # 1. 为每个电池单元生成 3 个子图（温度、电压、SOC）
    for i in range(cell_num):
        # 创建图像和子图（非交互式后端，无 GUI 窗口）
        fig, ax = plt.subplots(3, 1, figsize=(6, 4))
        
        # 绘制每个子图（温度、电压、SOC）
        for ii in range(3):
            ax[ii].plot(signal[i*3 + ii])  # 假设信号格式为 [cell0_temp, cell0_voltage, cell0_SOC, cell1_temp, ...]
            ax[ii].set_title(f"{cols_name[ii]} of Cell {i+1}")
            ax[ii].set_xlabel("Time (min)")
            ax[ii].set_ylabel(f"Value ({unit_name[ii]})")
        fig.tight_layout()  # 自动调整布局，避免标题重叠
        
        # 保存图像到内存缓冲区（BytesIO）
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.1  # 减少内边距，避免裁剪标题
        )
        plt.close(fig)  # 立即关闭图像，释放内存（关键！避免内存泄漏）
        
        # 转换为 base64 字符串并添加到列表
        buffer.seek(0)  # 移动到缓冲区开头
        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
        base64_images.append(f"data:image/png;base64,{base64_str}")

    return base64_images

def TimeSeriesImageCur(signal):
    fig, ax = plt.subplots(figsize=(6, 2))  # 单独一个子图
    ax.plot(signal[-1])  # 假设 signal[-1] 是模块电流数据
    ax.set_title("current of module")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Value (A)")
    fig.tight_layout()
    
    # 保存并转换为 base64
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=100,
        bbox_inches="tight",
        pad_inches=0.1
    )
    plt.close(fig)  # 关闭图像释放内存
    
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

def process_sample(idx, sample, instruct, client, OPENAI_MODEL):
    """
    线程任务函数：处理单个样本（API 调用 + 结果返回）
    :param idx: 样本索引（用于打印进度和排序结果）
    :param sample: 单个样本数据（sample_pick 中的元素）
    :param instruct2: 提示词前缀（原代码中的 instruct2）
    :param client: OpenAI 客户端实例（需确保线程安全）
    :param OPENAI_MODEL: 模型名称（如 "qwen-max"）
    :return: 元组 (idx, answers)：样本索引和 API 返回结果
    """
    try:
        # API 调用（同原循环内的逻辑）
        prompt = TimeSeriesImage(sample["data"])
        prompt_cur = TimeSeriesImageCur(sample["data"])
        prompt.append(prompt_cur)  # 添加电流图像
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user","content": [
                    {"type": "text", "text": instruct},
                    *[{"type": "image_url", "image_url": {"url": prompt[idx]}} for idx in range(17)],
                    ]}],
            temperature=0.3
            )
        answers = completion.choices[0].message.content.strip()
        print(f"样本 {idx} 处理完成：{answers[:200]}...")  # 打印部分结果（避免刷屏）
        return (idx, answers)  # 返回索引和结果（用于后续排序）
    
    except Exception as e:
        print(f"样本 {idx} 处理失败：{str(e)}")
        return (idx, None)  # 失败时返回 None


class PreprocessedQADataset:
    """仅保留数据预处理逻辑"""
    def __init__(self, jsonl_file):
        self.samples = self._load_data(jsonl_file)
        
    def _load_data(self, jsonl_file):
        current_file_path = Path(__file__)
        jsonl_file = current_file_path.parents[0] / "data" / jsonl_file
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

if __name__ == "__main__":
    # 从环境变量获取 API 密钥
    api_key = OPENAI_API_KEY

    # 创建评估器
    
    instruct1 = ("""Please analyze the following Time Series of battery module (consist of 16 series-connected LIB cells), and determine whether there are any anomalies in this time period.
    Here are some basic informations of the involved LIB cells: Nominal Voltage: 3.2V, Operating Voltage: 2.5~3.65V, Charging/Discharging Protocal: Constant Power, Ambient Temperature: 25degC.
    These basic informations indicate the expected normal operating ranges of LIB cells, which is important for your fault diagnosis tasks.
    You should conduct monitoring task through cluster-level (cluster behavior) and individual-level (each specific LIB cell).
    If a noticable anomaly is detected, you should further select the most probable anomaly type from these options:
    A. Temperature Anomaly in specific individual cell
    B. Temperature Anomaly of whole LIB module
    C. Voltage Anomaly in specific individual cell
    D. Voltage Anomaly of whole LIB module.
    Also, you should identify the beginning sample and the ending sample of the detected anomaly, e.g., "1th~50th sample"
    The output format: you should return your answers in the format with "The operation is <ans>normal</ans>." or "The operation is <ans>abnormal</ans>." in the begining, then the related analysis should be given.
    If the anomaly is detected, you should output its fault types, beginning time and the endding time.
    """)


    instruct2 = ("""Please analyze the following Time-Series plots of battery module to generate precise, actionable forecasts for charge and discharge cycle management.
                Your Primary Task:
1. State and Trend Analysis (Descriptive):
**Describe the current charging/discharging state of the battery module. 
**Report the recent state of charge (SOC) (Mean SOC, Maximal SOC, and Minimal SOC across all the cells within the module).
**Identify the overall trend of the module's SOC (e.g., increasing, decreasing, or stable).

2. Operational Forecast and Actionable Guidance:
**Based on the current SOC parameters and trend, predict the time until the module's critical SOC cutoff condition is met.
**Provide clear, practical operational advice regarding charge/discharge actions.

#### Specific Cutoff Conditions:
Charging Scenario: The charging porcess of LIB module must be terminated when the Maximal SOC reaches 97%.
Discharging Scenario: The discharging process of LIB module must be terminated when the Minimal SOC reaches 3%.
The maximal (minimal) SOC refer to the highest (lowest) SOC value among all individual battery cells within the module.

#### sampling interval: 
The sampling interval of involved battery time series is 1 minute.

There are some output examples:
Output Example 1 (Charging):
"Based on the observed time-series data, the battery module experiences the Charging Phase. At the most recent timestamp, the module’s Mean SOC is <ans>mean_soc</ans>%, the Maximum SOC is <ans>max_soc</ans>%, and the Minimum SOC is <ans>min_soc</ans>%. The overall SOC exhibits an upward trend. Our forecast indicates that the module's Maximum SOC will reach the 97% cutoff condition in approximately <ans>pred_time</ans> minutes. To prevent overcharging and ensure battery longevity, it is advised to terminate charging after <ans>pred_time</ans> minutes. Detailed Analysis Follows:..."

Output Example 2 (Discharging):
"Based on the observed time-series data, the battery module experiences the Discharging Phase. At the most recent timestamp, the module’s Mean SOC is <ans>mean_soc</ans>%, the Maximum SOC is <ans>max_soc</ans>%, and the Minimum SOC is <ans>min_soc</ans>%. The overall SOC exhibits a downward trend. Our forecast indicates that the module's Minimum SOC will reach the 3% cutoff condition in approximately <ans>pred_time</ans> minutes. To prevent deep discharge and irreversible degradation, it is advised to terminate discharging after <ans>pred_time</ans> minutes. Detailed Analysis Follows:..."

### Attention:
1. Output Format Specification: The final output must strictly adhere to the following structure, using the <ans></ans> tag to explicitly highlight all critical numerical data.
2. You should first present all the key conclusions, and then provide the detailed analysis.
3. The output answers should include a Detailed Analysis that fully describes the reasoning for your conclusions, which is crucial for operators to understand the basis of the operational decisions. Furthermore, provide any additional practical recommendations or safety warnings derived from the analysis.
4. The maximal, minimal, average SOC refer to the highest, lowest, and average SOC value among all individual battery cells within the module.
                """)
    
    dataset = PreprocessedQADataset("template_mana_qwen3-32b_xs_20251021.jsonl")
    # dataset = PreprocessedQADataset("template_ad_qa_dsv3_10.jsonl")
    
    # 测试第一个样本
    input_texts = []
    true_values = []
    sample_pick = dataset.samples
    
    sample_pick = [sample_pick[0]] + [sample_pick[0]] + [sample_pick[0]] + [sample_pick[0]] + [sample_pick[0]]
    
    max_workers = 10 # 最大并发数（根据 API 速率限制调整，如 5~10）
    print(f"a total of {len(sample_pick)} samples")
    results = [None] * len(sample_pick)  # 预分配结果列表（按样本索引存储）
    hand_sample = [0]
    # 2. 创建线程池并提交任务
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有样本任务（每个样本对应一个 future）
        futures = [
            executor.submit(
                process_sample,
                idx=i,
                sample=sample_pick[i],
                instruct=instruct2,  # 原循环中的 instruct2（假设已定义）
                client=client,  # 原循环中的 client（假设已定义，需线程安全）
                OPENAI_MODEL=OPENAI_MODEL  # 原循环中的模型名称（假设已定义）
            )
            for i in range(len(sample_pick)) # 原循环范围：range(len(sample_pick)-1)
        ]
        
        # 3. 获取完成的任务结果并按索引存储（保证结果顺序与样本顺序一致）
        for future in as_completed(futures):
            idx, answers = future.result()  # 获取任务返回的 (idx, answers)
            results[idx] = answers  # 按样本索引存入结果列表
        
    # 4. 打印所有结果（按样本顺序）
    result_list = []
    for idx, answers in enumerate(results):
        if answers is not None:
            # print(f"样本 {idx} 结果：\n{answers}\n")
            # match = re.search(r'<ans>([-+]?\d*\.?\d+)</ans>', answers)
            match = re.search(r'<ans>(.*?)</ans>', answers)
            if match:
                first_value = match.group(1)
                result_list.append(first_value)
            else:
                result_list.append(-1)
        else:
            print(f"样本 {idx} 处理失败\n")
    
    result_list = np.array(result_list)

        # 4. 打印分隔符（原循环中的 "-------------------------next sample-----------------------"）
