<div align="center">

# TimeSeries2Report prompting enables adaptive large language model management of lithium-ion batteries

_This is the official implementation of **TimeSeries2Report (TS2R)** in the paper "TimeSeries2Report prompting enables adaptive large language model management of lithium-ion batteries"_
[![Paper](https://img.shields.io/badge/Paper-PDF-%2300B4FF)](https://arxiv.org/abs/2512.16453)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
</div>

<p align="center">
⚡ &nbsp;<a href="#-key-features">Key Features</a>
| &nbsp;<a href="#-TS2R-framework">Framework</a>
| &nbsp;<a href="#-quick-start">Quick Start</a>
| &nbsp;<a href="#-citation">Citation</a>
</p>

---
## ⚡ Key Features
✅ **Zero-shot time-series analysis** - Enables LLMs to interpret battery data without retraining  
✅ **Semantic enrichment** - Converts raw sensor data into structured natural language reports  
✅ **Multi-task support** - SOC prediction, anomaly detection, and charge management  
✅ **Expert-level performance** - Achieves >90% factual accuracy in benchmark tests  
✅ **Model-agnostic** - Works with diverse LLMs (Qwen, DeepSeek, ChatGLM, Llama)  
✅ **No training required** - Lightweight and ready for deployment  

**TS2R-integrated LLMs achieve expert-level decision quality without model retraining or architecture modification, establishing a practical path for adaptive LLM-driven battery intelligence.**

---

## TS2R Framework
The TimeSeries2Report framework bridges low-level sensor signals with high-level contextual insights through:
1. **Time-series Segmentation**  
   Splits time-series into 10-timestamp slices and extracts temporal attributes  
   ```python
   # Pseudo-code for segmentation
   segments = [data[i:i+window] for i in range(0, len(data), step)]
   ```
2. **Descriptor Assignment**
   Converts numerical features to descriptors (e.g., “increasing”, “stable”) using rule-based mapping
   Trend: increasing → “Voltage shows steady upward progression”
   
3. **Report Synthesis**
   LLM integrates descriptors into coherent operational reports
Prompt:
"You are a battery expert. Describe cell behavior based on: ..." (see `prompting_engineering.py` for details)

![TS2R Framework Architecture](figs/figure_method.svg)
<p align="center"><em>Figure: TS2R framework architecture</em></p>

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Access
#### 2.1. Remote API calls
If you do not have a deployed LLM on your device, you can utilize API calls from mainstream providers such as [DeepSeek](https://platform.deepseek.com/api_keys) and [Qwen](https://bailian.console.aliyun.com/?spm=5176.12818093_47.aillm.1.3be92cc9v6vQfH&tab=model&scm=20140722.S_%E5%9C%A8%E5%93%AA%E9%87%8C%E8%83%BD%E8%8E%B7%E5%8F%96%E9%98%BF%E9%87%8C%E4%BA%91%E7%99%BE%E7%82%BC%E5%B9%B3%E5%8F%B0%E7%9A%84APIkey._.RL_%E5%9C%A8%E5%93%AA%E9%87%8C%E8%83%BD%E8%8E%B7%E5%8F%96%E9%98%BF%E9%87%8C%E4%BA%91%E7%99%BE%E7%82%BC%E5%B9%B3%E5%8F%B0%E7%9A%84APIkey-LOC_aillm-OR_chat-V_3-RC_llm#/model-market).

#### 2.2. Local API calls
For local API calls, make sure that you have deployed the large model on the server. You need to set up the corresponding API interface by running the `local_api.py` script, which we have provided in our `utils` framework.
```bash
python your_file_path/local_api.py
```

### 3. Prepare Datasets

Place battery time-series data in the `./time_series_data` directory. We have already prepared the necessary data files to ensure that the code is executable. You can replace these files with your own data.

### 4. Battery time-series report generation through simple API call

If utilizing remote API calls, run `TS2R_generation_main.py` with the following parameter settings to generate reports for time-series data in ZJU dataset:

```python
# set utilized LLM
llm_list = ['qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b-instruct-2507', 'deepseek-chat', 'glm-4.5']
OPENAI_MODEL = llm_list[0] 

# set TOTAL_CNT (line 280) to decide the processing and generation number of time-seires reports
TOTAL_CNT = 100 

# revise the data reading method (lines 288~291) according your own data format (take json file as example, which we have already prepared) 
dataset_name = "human-description20251210.json"
dataset = PreprocessedDataset(dataset_name)
data_v_win_train, data_c_win_train = prre.read_data_from_json(dataset)
# size of data_v_win_train: (sequence_length, variable_num, battery_num)
# size of data_c_win_train: (sequence_length, 1) (for shared-variable like current for series-connected batteries)

# set the output path (line 299)
OUTPUT_DATASET = f'{dir_path}\\report_{OPENAI_MODEL}_zju_{TOTAL_CNT}_{timestamp[:8]}.json'
```

If employing local API calls, run `TS2R_generation_local_main.py` with the following parameter settings to generate reports for time-series data in ZJU dataset:

```python

# set TOTAL_CNT (line 334) to decide the processing and generation number of time-seires reports
TOTAL_CNT = 100 

# revise the data reading method (lines 342~344) according your own data format (take json file as example, which we have already prepared) 
dataset_name = "human-description20251210.json"
dataset = PreprocessedDataset(dataset_name)
data_v_win_train, data_c_win_train = prre.read_data_from_json(dataset)
# size of data_v_win_train: (sequence_length, variable_num, battery_num)
# size of data_c_win_train: (sequence_length, 1) (for shared-variable like current for series-connected batteries)

# set the output path (line 351)
OUTPUT_DATASET = f'{dir_path}\\report_zju_{TOTAL_CNT}_{timestamp[:8]}.json'
```
Similarly, you can also generate reports for MIT/TJU dataset using `TS2R_generation_mit_main.py`/`TS2R_generation_tju_main.py`, and `TS2R_generation_local_mit_main.py`/`TS2R_generation_local_tju_main.py` through API calls.

### 5. LLM-driven battery downstream task application --- SOC prediction

run `downstream_SOC_prediction.py` with the following parameter settings:

```python
# set utilized LLM (line 36)
OPENAI_MODEL = "qwen3-14b"

# set the path to save the SOC precition results (lines 123~127)
current_file_path = Path(__file__)
test_set = '20251210'
dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"

# add the path where TS2R report is stored (line 129)
dataset_name = current_file_path.parents[0] / "output_data" / f"report_pred_{OPENAI_MODEL}_zju_100_{test_set}.json"
```

### 6. LLM-driven battery downstream task application --- anomaly detection

run `downstream_anomaly_detection.py` with the following parameter settings:

```python
# set utilized LLM (line 39)
OPENAI_MODEL = "qwen3-14b"

# set the path to save the anomaly detection results (lines 162~166)
current_file_path = Path(__file__)
test_set = '20251210'
dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"

# add the path where TS2R report is stored (lines 171~175)
# for normally operation cases
dataset_name = current_file_path.parents[0] / "output_data" / f"report_{OPENAI_MODEL}_zju_100_{test_set}.json"
    
# for abnormally operation cases
# dataset_name = current_file_path.parents[0] / "output_data" / f"report_ano_{OPENAI_MODEL}_zju_3_{test_set}.json"
```

### 7. LLM-driven battery downstream task application --- charging/discharging management

run `downstream_charge_management.py` with the following parameter settings:

```python
# set utilized LLM (line 36)
OPENAI_MODEL = "qwen3-14b"

# set the path to save the charging/discharging management results (lines 151~155)
current_file_path = Path(__file__)
test_set = '20251210'
dir_path = current_file_path.parents[0] / f"downstream_result_{test_set}"

# add the path where TS2R report is stored (line 157)
dataset_name = current_file_path.parents[0] / "output_data" / f"report_{OPENAI_MODEL}_zju_100_{test_set}.json"
```

## 🔗 Citation

If you find this code useful for your research, please cite

- [[Jiayang Yang, Chunhui Zhao, Martin Guay, Zhixing Cao. TimeSeries2Report prompting enables adaptive large language model management of lithium-ion batteries. arXiv preprint arXiv:2512.16453, 2025.](https://arxiv.org/abs/2512.16453)]

```BibTeX
@article{yang2025timeseries2report,
  title={TimeSeries2Report prompting enables adaptive large language model management of lithium-ion batteries},
  author={Yang, Jiayang and Guay, Martin and Cao, Zhixing and Zhao, Chunhui},
  journal={arXiv preprint arXiv:2512.16453},
  year={2025}
}
```
