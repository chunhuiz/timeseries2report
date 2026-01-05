# -*- coding: utf-8 -*-
"""
Prompt Engineering Module for Battery Time-Series Report Generation

This module provides functions for:
- Cleaning and formatting text prompts
- Generating structured reports of battery time-series data
- Validating and processing JSON responses from LLMs
- Converting between different description formats

Key Functions:
- clean_manual_linebreaks: Format text by removing unnecessary line breaks
- restore_clu_text: Generate cluster-level battery reports
- check_clu_des: Validate and format cluster reports 
- restore_ind_text: Generate individual cell reports

"""

import re
import json

def clean_manual_linebreaks(text):
    """Remove manual line breaks while preserving paragraph structure
    
    Processes text to:
    - Remove leading whitespace from each line
    - Replace single newlines with spaces
    - Consolidate multiple spaces
    - Preserve blank lines as paragraph breaks
    """
    # Remove indentation from each line
    dedented = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
    
    # Replace single newlines (not at paragraph end) with spaces
    cleaned = re.sub(r'(?<!\n)\n(?!\n|\Z)', ' ', dedented)
    
    # Collapse multiple spaces into single space
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    
    # Normalize multiple newlines to single paragraph break
    return re.sub(r'\n{2,}', '\n', cleaned)

def restore_clu_text(clu_json, description_begin_period, description_end_period, description_object):
    """Generate comprehensive cluster-level battery description from JSON data
    
    Constructs a detailed report including:
    - Overall operation summary
    - Cluster perspective analysis
    - Voltage, temperature and SOC characteristics
    - Current characteristics
    """
    if type(clu_json) is str:
        clu_data = json.loads(clu_json)
    else:
        clu_data = clu_json

    # Build introduction section
    description_text = []
    description_intro = (f"""# Lithium-ion battery Time-Series Descriptive Report\n
    - **Description Scope**: This report describes the time series of a lithium-ion battery (LIB) module, which is organized as follows:
    First, the overall operation conclusion of the battery module is described, including the charging and discharging stages
    it experienced, the duration of each stage, the initial/final charge states, and the inconsistency situations among internal
    cells. Then the detailed operation states of the battery module are provided from both the cluster and individual perspectives.
    When describing the cluster states, the average operating state of all the battery cells within the module and the inconsistency
    between the cells will be described, respectively. The average operating state includes four aspects:
    trend, transition events, fluctuation, and outlier phenomenon. The inconsistency between the cells will be evaluated
    using dispersion indicators such as standard deviation and Shannon entropy. When describing the individual situation
    of each battery cell within the module, the texts will be given from two dimensions: shape and value.
    The shape description includes: change trend and rate, transition events, fluctuation, and outliers;
    The value description includes key statistical indicators such as mean, variance, mode, and the initial/final values.\n
    ## I. Overall operation conclusion:\n
    1. **Overall experienced operation**: {clu_data['Overall_operation']['overall_operation']}\n
    2. **Overall inconsistency situations**: {clu_data['Overall_operation']['overall_inconsistency']}
    """)
    description_text.append(description_intro)
    description_text.append("## II. **Cluster Perspective**:\n")
    description_text.append("""Within a certain LIB module,
    the signals comprising current, voltage, temperature, and State of Charge (SOC)
    of the series-connected battery cells are collected from the Battery Management System (BMS)
    and organized into time series. For series-connected battery cells, the current is shared,
    while the voltage, temperature, and SOC vary among cells. Therefore, for voltage, temperature,
    and SOC, the cluster perspective describes both the average operating state of all the LIB cells
    and their inconsistency. While for current, it only describes the uniform current of the module.
    To be more specific, the average operating state includes the trend, transition events, fluctuation, and outlier phenomenon,
    while the inconsistency is evaluated using dispersion indicators including standard deviation, Shannon entropy, and max. & min. values.
""")

    # Process voltage, temperature and SOC characteristics
    vara_list = ["Voltage", "Temperature", "SOC"]
    
    if "outliers" in clu_data['Current']:
        cur_out = clu_data['Current']["outliers"]
    if "outlier_phenomenon" in clu_data['Current']:
        cur_out = clu_data['Current']["outlier_phenomenon"]
        
    for i in range(3):
        var = vara_list[i]
        avg = clu_data[var]['average']
        max_ = clu_data[var]['maximum'] 
        min_ = clu_data[var]['minimum']
        std = clu_data[var]['standard_deviation']
        entropy = clu_data[var]['shannon_entropy']

        description_detail = (f"""
    ### 1.{i+1}. {var} Characteristics:\n
    #### 1.{i+1}.1. Overall Operation:\n
    ##### 1.{i+1}.1.1. Average {var} of the LIB Module:\n
    - **Trend**:{avg['trend']}\n
    - **Transition Events**:{avg['transitions'] if 'transitions' in avg else avg['transition_events']}\n
    - **Fluctuation**:{avg['fluctuation']}\n
    - **Outlier Phenomenon**:{avg['outliers']}\n
    - **mean and std**: {avg['mean_and_std']}\n
    - **initial/final values**: {avg['initial_final']}\n
    ##### 1.{i+1}.1.2. Maximum {var} of the LIB Module:\n
    - **Trend**:{max_['trend']}\n
    - **Transition Events**:{max_['transitions'] if 'transitions' in max_ else max_['transition_events']}\n
    - **mean and std**: {max_['mean_and_std']}\n
    - **initial/final values**: {max_['initial_final']}\n 
    ##### 1.{i+1}.1.3. Minimum {var} of the LIB Module:\n
    - **Trend**:{min_['trend']}\n
    - **Transition Events**:{min_['transitions'] if 'transitions' in min_ else min_['transition_events']}\n
    - **mean and std**: {min_['mean_and_std']}\n
    - **initial/final values**: {min_['initial_final']}\n

    #### 1.{i+1}.2. Inconsistency of {var} among Cells:\n
    ##### 1.{i+1}.2.1. Standard Deviation Situation:\n
    - **amplitude**: {std['amplitude']}\n
    - **fluctuation**: {std['fluctuation']}\n
    - **mean and std**: {std['mean_and_std']}\n
    #### 1.{i+1}.2.2. Shannon Entropy Situation:\n
    - **amplitude**: {entropy['amplitude']}\n
    - **fluctuation**: {entropy['fluctuation']}\n
    - **mean and std**: {entropy['mean_and_std']}\n
    """)
        description_text.append(description_detail)

    # Add current characteristics
    description_cur = f"""
    ### 1.4. Current Characteristics:\n
    - **Trend**:{clu_data['Current']['trend']}\n
    - **Transition Events**:{clu_data['Current']['transition_events']}\n
    - **Fluctuation**:{clu_data['Current']['fluctuation']}\n
    - **Outlier Phenomenon**:{cur_out}\n
    - **mean and std**: {clu_data['Current']['mean_and_std']}\n
    - **initial/final values**: {clu_data['Current']['initial_final']}
    """
    description_text.append(description_cur)
    
    return clean_manual_linebreaks("\n".join(description_text))

def check_clu_des(clu_json):
    """Validate and format cluster description JSON into readable text
    
    Creates a condensed version of cluster description focusing on key metrics
    """
    if type(clu_json) is str:
        clu_data = json.loads(clu_json)
    else:
        clu_data = clu_json

    description_intro = f"""
    1. **Overall experienced operation**: {clu_data['Overall_operation']['overall_operation']}\n
    2. **Overall inconsistency situations**: {clu_data['Overall_operation']['overall_inconsistency']}
    """
    
    description_text = [description_intro]
    vara_list = ["Voltage", "Temperature", "SOC"]
    if "outliers" in clu_data['Current']:
        cur_out = clu_data['Current']["outliers"]
    if "outlier_phenomenon" in clu_data['Current']:
        cur_out = clu_data['Current']["outlier_phenomenon"]
        
    for i in range(3):
        var = vara_list[i]
        avg = clu_data[var]['average']
        max_ = clu_data[var]['maximum']
        min_ = clu_data[var]['minimum']
        std = clu_data[var]['standard_deviation']
        entropy = clu_data[var]['shannon_entropy']

        description_detail = (f"""
    ### 1.{i+1}. {var} Characteristics:\n
    #### 1.{i+1}.1. Overall Operation:\n
    ##### 1.{i+1}.1.1. Average {var} of the LIB Module:\n
    - **Trend**:{avg['trend']}\n
    - **Transition Events**:{avg['transitions'] if 'transitions' in avg else avg['transition_events']}\n
    - **Fluctuation**:{avg['fluctuation']}\n
    - **Outlier Phenomenon**:{avg['outliers']}\n
    - **mean and std**: {avg['mean_and_std']}\n
    - **initial/final values**: {avg['initial_final']}\n
    ##### 1.{i+1}.1.2. Maximum {var} of the LIB Module:\n
    - **Trend**:{max_['trend']}\n
    - **Transition Events**:{max_['transitions'] if 'transitions' in max_ else max_['transition_events']}\n
    - **mean and std**: {max_['mean_and_std']}\n
    - **initial/final values**: {max_['initial_final']}\n 
    ##### 1.{i+1}.1.3. Minimum {var} of the LIB Module:\n
    - **Trend**:{min_['trend']}\n
    - **Transition Events**:{min_['transitions'] if 'transitions' in min_ else min_['transition_events']}\n
    - **mean and std**: {min_['mean_and_std']}\n
    - **initial/final values**: {min_['initial_final']}\n

    #### 1.{i+1}.2. Inconsistency of {var} among Cells:\n
    ##### 1.{i+1}.2.1. Standard Deviation Situation:\n
    - **amplitude**: {std['amplitude']}\n
    - **fluctuation**: {std['fluctuation']}\n
    - **mean and std**: {std['mean_and_std']}\n
    #### 1.{i+1}.2.2. Shannon Entropy Situation:\n
    - **amplitude**: {entropy['amplitude']}\n
    - **fluctuation**: {entropy['fluctuation']}\n
    - **mean and std**: {entropy['mean_and_std']}\n
    """)
        description_text.append(description_detail)

    description_cur = f"""
    ### 1.4. Current Characteristics:\n
    - **Trend**:{clu_data['Current']['trend']}\n
    - **Transition Events**:{clu_data['Current']['transition_events']}\n
    - **Fluctuation**:{clu_data['Current']['fluctuation']}\n
    - **Outlier Phenomenon**:{cur_out}\n
    - **mean and std**: {clu_data['Current']['mean_and_std']}\n
    - **initial/final values**: {clu_data['Current']['initial_final']}
    """
    description_text.append(description_cur)
    
    return clean_manual_linebreaks("\n".join(description_text))

def pick_ind_data(data, idx):
    """Extract individual cell data from JSON structure
    
    Handles both raw JSON strings and parsed dictionaries
    """
    if "cells_info" in data.keys():
        ind_data_pick_temp = data["cells_info"]
    else:
        ind_data_pick_temp = data["raw_response"]
    if type(ind_data_pick_temp) is str:
        ind_data_pick = json.loads(ind_data_pick_temp)[idx]
    else:
        ind_data_pick = ind_data_pick_temp[idx]
    
    return ind_data_pick

def restore_ind_text(ind_json1, ind_json2, ind_json3, ind_json4, cell_num=16):
    """Generate detailed individual cell descriptions from JSON data
    
    Creates reports for each cell covering:
    - Voltage characteristics
    - Temperature characteristics  
    - SOC characteristics
    """
    description_text = []
    description_intro = ("""## III. **Individual Perspective**:\n
    The individual perspective describes the operating states of each battery cell
    within the LIB module. The descriptions are provided for the voltage, temperature, 
    and SOC of each cell from two-fold aspects: shape-related and statistics-related.
    The shape-related descriptions focus on change trend and rate, transition events, fluctuation,
    and outliers of concerned signals, while the statistics-related descriptions include key statistical
    indicators including mean and variance (for stable phase) and the initial/final values 
    (for increasing or decreasing phase).
    """)
    description_text.append(description_intro)

    # Process each of the 4 JSON inputs
    ind_data1 = json.loads(ind_json1) if type(ind_json1) is str else ind_json1
    ind_data2 = json.loads(ind_json2) if type(ind_json2) is str else ind_json2 
    ind_data3 = json.loads(ind_json3) if type(ind_json3) is str else ind_json3
    ind_data4 = json.loads(ind_json4) if type(ind_json4) is str else ind_json4

    for i in range(cell_num):
        # Select appropriate data source based on cell index
        if i < 4:
            ind_data_pick = pick_ind_data(ind_data1, i)
        elif i < 8:
            ind_data_pick = pick_ind_data(ind_data2, i-4)
        elif i < 12:
            ind_data_pick = pick_ind_data(ind_data3, i-8)
        else:
            ind_data_pick = pick_ind_data(ind_data4, i-12)

        # Extract metrics with fallback for key variations
        if "fluctuation" in ind_data_pick['voltage']:
                vol_flu = ind_data_pick['voltage']["fluctuation"]
        if "flucutation" in ind_data_pick['voltage']:
            vol_flu = ind_data_pick['voltage']["flucutation"]
        if "fluctuation" in ind_data_pick['temperature']:
            tem_flu = ind_data_pick['temperature']["fluctuation"]
        if "flucutation" in ind_data_pick['temperature']:
            tem_flu = ind_data_pick['temperature']["flucutation"]
        if "fluctuation" in ind_data_pick['soc']:
            soc_flu = ind_data_pick['soc']["fluctuation"]
        if "flucutation" in ind_data_pick['soc']:
            soc_flu = ind_data_pick['soc']["flucutation"]

        description_ind = (f"""
    ### 2.{i+1}. LIB Cell #{i+1}:\n
    #### 2.{i+1}.1. Voltage:\n
    - **trend**: {ind_data_pick['voltage']['trend']}\n
    - **transition events**: {ind_data_pick['voltage'].get('transition') or ind_data_pick['voltage'].get('transition events')}\n
    - **fluctuation**: {vol_flu}\n
    - **outliers**: {ind_data_pick['voltage']['outliers']}\n
    - **mean and std (for stable phase)**: {ind_data_pick['voltage'].get('mean_and_std') or ind_data_pick['voltage'].get('mean and std (for stable phase)')}\n
    - **initial/final values (for increasing or decreasing phase)**: {ind_data_pick['voltage'].get('initial_final') or ind_data_pick['voltage'].get('initial/final values (for increasing or decreasing phase)')}\n
    
    #### 2.{i+1}.2. Temperature:\n
    - **trend**: {ind_data_pick['temperature']['trend']}\n
    - **transition events**: {ind_data_pick['temperature'].get('transition') or ind_data_pick['temperature'].get('transition events')}\n
    - **fluctuation**: {tem_flu}\n
    - **outliers**: {ind_data_pick['temperature']['outliers']}\n
    - **mean and std (for stable phase)**: {ind_data_pick['temperature'].get('mean_and_std') or ind_data_pick['temperature'].get('mean and std (for stable phase)')}\n
    - **initial/final values (for increasing or decreasing phase)**: {ind_data_pick['temperature'].get('initial_final') or ind_data_pick['temperature'].get('initial/final values (for increasing or decreasing phase)')}\n
    
    #### 2.{i+1}.3. SOC:\n
    - **trend**: {ind_data_pick['soc']['trend']}\n
    - **transition events**: {ind_data_pick['soc'].get('transition') or ind_data_pick['soc'].get('transition events')}\n
    - **fluctuation**: {soc_flu}\n
    - **outliers**: {ind_data_pick['soc']['outliers']}\n
    - **mean and std (for stable phase)**: {ind_data_pick['soc'].get('mean_and_std') or ind_data_pick['soc'].get('mean and std (for stable phase)')}\n
    - **initial/final values (for increasing or decreasing phase)**: {ind_data_pick['soc'].get('initial_final') or ind_data_pick['soc'].get('initial/final values (for increasing or decreasing phase)')}\n
    """)
        description_text.append(description_ind)
    
    return clean_manual_linebreaks("\n".join(description_text))

def check_ind_des(ind_json1, cell_num=4, dataset_name='zju'):
    """Validate and format individual cell descriptions from JSON data
    
    Processes JSON containing descriptions for multiple battery cells and:
    - Validates the structure
    - Formats into consistent text output
    - Handles variations in field names
    - Cleans up formatting
    
    Args:
        ind_json1: JSON string or dict containing cell descriptions
        cell_num: Number of cells to process (default: 4)
        
    Returns:
        Formatted text report with all cell descriptions
    """
    description_text = []
    
    # Parse JSON if input is string
    if type(ind_json1) is str:
        ind_data1 = json.loads(ind_json1)
    else:
        ind_data1 = ind_json1

    if dataset_name == 'zju':
        # Process each specified cell
        for i in range(cell_num):
            # Extract data for current cell
            ind_data_pick = pick_ind_data(ind_data1, i)
            
            # Handle variations in 'fluctuation' key spelling
            vol_flu = ind_data_pick['voltage'].get("fluctuation", 
                    ind_data_pick['voltage'].get("flucutation", ""))
            tem_flu = ind_data_pick['temperature'].get("fluctuation", 
                    ind_data_pick['temperature'].get("flucutation", ""))
            soc_flu = ind_data_pick['soc'].get("fluctuation", 
                    ind_data_pick['soc'].get("flucutation", ""))
                    
            # Format cell description with fallbacks for alternate field names
            description_ind = f"""
                                ### 2.{i+1}. LIB Cell #{i+1}:\n
                                #### 2.{i+1}.1. Voltage:\n
                                - **trend**: {ind_data_pick['voltage']['trend']}\n
                                - **transition events**: {ind_data_pick['voltage'].get('transition', 
                                                        ind_data_pick['voltage'].get('transition events', ''))}\n
                                - **fluctuation**: {vol_flu}\n
                                - **outliers**: {ind_data_pick['voltage']['outliers']}\n
                                - **mean and std**: {ind_data_pick['voltage'].get('mean_and_std', 
                                                ind_data_pick['voltage'].get('mean and std (for stable phase)', ''))}\n
                                - **initial/final values**: {ind_data_pick['voltage'].get('initial_final', 
                                                        ind_data_pick['voltage'].get('initial/final values (for increasing or decreasing phase)', ''))}\n

                                #### 2.{i+1}.2. Temperature:\n
                                - **trend**: {ind_data_pick['temperature']['trend']}\n
                                - **transition events**: {ind_data_pick['temperature'].get('transition', 
                                                        ind_data_pick['temperature'].get('transition events', ''))}\n
                                - **fluctuation**: {tem_flu}\n
                                - **outliers**: {ind_data_pick['temperature']['outliers']}\n
                                - **mean and std**: {ind_data_pick['temperature'].get('mean_and_std', 
                                                ind_data_pick['temperature'].get('mean and std (for stable phase)', ''))}\n
                                - **initial/final values**: {ind_data_pick['temperature'].get('initial_final', 
                                                        ind_data_pick['temperature'].get('initial/final values (for increasing or decreasing phase)', ''))}\n

                                #### 2.{i+1}.3. SOC:\n
                                - **trend**: {ind_data_pick['soc']['trend']}\n
                                - **transition events**: {ind_data_pick['soc'].get('transition', 
                                                        ind_data_pick['soc'].get('transition events', ''))}\n
                                - **fluctuation**: {soc_flu}\n
                                - **outliers**: {ind_data_pick['soc']['outliers']}\n
                                - **mean and std**: {ind_data_pick['soc'].get('mean_and_std', 
                                                ind_data_pick['soc'].get('mean and std (for stable phase)', ''))}\n
                                - **initial/final values**: {ind_data_pick['soc'].get('initial_final', 
                                                        ind_data_pick['soc'].get('initial/final values (for increasing or decreasing phase)', ''))}\n
                                """
            description_text.append(description_ind)
    
    elif dataset_name == 'mit':
        for i in range(cell_num):
            ind_data_pick = pick_ind_data(ind_data1,i)
            if "discharge capacity" in ind_data_pick:
                est_var = ["voltage", "temperature", "charge capacity", "discharge capacity", "current"]
            elif "discharge_capacity" in ind_data_pick:
                est_var = ["voltage", "temperature", "charge_capacity", "discharge_capacity", "current"]
            
            description_text.append(f"LIB Cell #{i+1}:")

            for vara in est_var:
                if "fluctuation" in ind_data_pick[vara]:
                    vol_flu = ind_data_pick[vara]["fluctuation"]
                if "flucutation" in ind_data_pick[vara]:
                    vol_flu = ind_data_pick[vara]["flucutation"]
                description_ind = (f"""
            {vara}:\n
            - **trend**: {ind_data_pick[vara]['trend']}\n
            - **transition events**: {ind_data_pick[vara].get('transition') or ind_data_pick[vara].get('transition events')}\n
            - **fluctuation**: {vol_flu}\n
            - **outliers**: {ind_data_pick[vara]['outliers']}\n
            - **mean and std (for stable phase)**: {ind_data_pick[vara].get('mean_and_std') or ind_data_pick[vara].get('mean and std (for stable phase)')}\n
            - **initial/final values (for increasing or decreasing phase)**: {ind_data_pick[vara].get('initial_final') or ind_data_pick[vara].get('initial/final values (for increasing or decreasing phase)')}\n
            """)
                description_text.append(description_ind)
    elif dataset_name == 'tju':
        for i in range(cell_num):
            ind_data_pick = pick_ind_data(ind_data1,i)
            if "discharge capacity" in ind_data_pick:
                est_var = ["voltage", "charge capacity", "discharge capacity", "current"]
            elif "discharge_capacity" in ind_data_pick:
                est_var = ["voltage", "charge_capacity", "discharge_capacity", "current"]

            description_text.append(f"LIB Cell #{i+1}:")
            
            for vara in est_var:
                if "fluctuation" in ind_data_pick[vara]:
                    vol_flu = ind_data_pick[vara]["fluctuation"]
                if "flucutation" in ind_data_pick[vara]:
                    vol_flu = ind_data_pick[vara]["flucutation"]
                description_ind = (f"""
            {vara}:\n
            - **trend**: {ind_data_pick[vara]['trend']}\n
            - **transition events**: {ind_data_pick[vara].get('transition') or ind_data_pick[vara].get('transition events')}\n
            - **fluctuation**: {vol_flu}\n
            - **outliers**: {ind_data_pick[vara]['outliers']}\n
            - **mean and std (for stable phase)**: {ind_data_pick[vara].get('mean_and_std') or ind_data_pick[vara].get('mean and std (for stable phase)')}\n
            - **initial/final values (for increasing or decreasing phase)**: {ind_data_pick[vara].get('initial_final') or ind_data_pick[vara].get('initial/final values (for increasing or decreasing phase)')}\n
            """)
                description_text.append(description_ind)
    else:
        raise{'no such dataset!'}
    
    # Combine all descriptions and clean formatting
    return clean_manual_linebreaks("\n".join(description_text))


def expert_prompt_for_report():
    """Generate expert prompt templates for lithium-ion battery time-series analysis reports.
    
    Constructs a set of standardized templates for describing battery module and cell characteristics,
    including voltage, temperature, SOC, and current parameters. The templates are designed to be
    filled with specific time-series analysis results.
    
    Returns:
        list: A list of 5 prompt strings, each combining a system prompt with a template:
            1. Main module analysis template
            2. Cell analysis template for cells 1-4
            3. Cell analysis template for cells 5-8 
            4. Cell analysis template for cells 9-12
            5. Cell analysis template for cells 13-16
             
        Each prompt includes:
        - Detailed instructions for the analyst
        - Structured output templates with placeholders
        - JSON schema specifications for the expected output format
    """    
    
    # System-level prompts
    SYSTEM_PROMPT1 = ("""
    You are an expert specializing in time-series signal analysis and maintenance for lithium-ion batteries.
    Your task is to complete the following "Lithium-ion battery Time-Series Descriptive Report" template based on the specific information provided by the user.
    Attention: 
    1. Please strictly follow the text template's logic, fill in the corresponding part within the curly braces, and convert them into JSON format.
    2. When you fill {overall_operation}, you are asked to describe the charging/discharging behavior of LIB module according the the module Current value (positive current means charging, negetive current means discharging, near zero means idle),
    e.g., "From 1 to 50 samples, the LIB module undergoes charging, from 50 to 100 samples, it undergoes discharging."
    and when you fill {overall_inconsistency}, you are asked to highlight the noticeable inconsistency situation of LIB module,
    e.g., "In most of period, LIB cells have well consistency. However, a moderate STD is observed from SOC at samples 1 to 20."
    3. The description should be in concise language. particularly, for descriptions regarding "Overall Operational Situation",
    **in "trend", please output "increase (1~50th time), stable (50~100th time)" rather than "increase from 29.0°C to 30.0°C from 1 to 50th sample, stable from 30.0°C to 31.0°C from 51 to 100th sample."
    **in "mean and std", ONLY the mean and std regarding the "stable" phases are considered, please output like "stable at 3.2V, std is 0.0043 (1~50th time)." for ALL the stable phases respectively.
    **in "initial/final values", ONLY initial/final values of the "increase" (or "descrease") phases are considered, please output like "3.2V to 3.3V (1~50th time)." for ALL the increase (or descrease) phases respectively.
                     
    Instruct Example: when filling in the part of {average_voltage_trend}, you will find "increase from 3.2V to 4.1V" in the user's information.
    then you should fill in {average_voltage_trend} with "The average voltage of the LIB module shows an increasing trend, rising from 3.2V to 4.1V over the described period."
                                          """)
                                          
    text_template1 = ("""
    Output Template Example:\n
    Lithium-ion battery Time-Series Descriptive Report\n
    Description Period: {description_begin_period} ~ {description_end_period}\n
    Description Object: {description_object}\n
                     
    I. Overall operation conclusion:
    1. **Overall experienced operation**:  {overall_operation}
    2. **Overall inconsistency situations**: {overall_inconsistency}
                     
    II. **Cluster Perspective**:\n
    1.1. Voltage Characteristics:\n
    1.1.1. Overall Operational Situation:\n
    a). average voltage of the LIB module:\n
    **Trend**: {average_voltage_trend}\n
    **Transition Events**: {average_voltage_transition_events}\n
    **Fluctuation**: {average_voltage_fluctuation}\n
    **Outlier Phenomenon**: {average_voltage_outlier_phenomenon}\n
    **mean and std (for stable phase)**: {average_voltage_mean_std}\n
    **initial/final values (for increasing or decreasing phase)**: {average_voltage_initial_final}\n
                     
    b). maximum voltage of the LIB module:\n
    **Trend**: {max_voltage_trend}\n
    **Transition Events**: {max_voltage_transition_events}\n
    **Outlier Phenomenon**: {max_voltage_outlier_phenomenon}\n
    **mean and std (for stable phase)**: {max_voltage_mean_std}\n
    **initial/final values (for increasing or decreasing phase)**: {max_voltage_initial_final}\n
    
    c). minimum voltage of the LIB module:\n
    **Trend**: {min_voltage_trend}\n
    **Transition Events**: {min_voltage_transition_events}\n
    **Outlier Phenomenon**: {min_voltage_outlier_phenomenon}\n
    **mean and std (for stable phase)**: {min_voltage_mean_std}\n
    **initial/final values (for increasing or decreasing phase)**: {min_voltage_initial_final}\n
                     
    1.1.2. Inconsistency of Voltage among Cells:\n
    a) Standard Deviation Situation:\n
    **Amplitude level**: {voltage_std_amplitude}\n
    **fluctuation level**: {voltage_std_fluctuation}\n
    **mean and std**: {voltage_std_mean_std}\n
    
    b) Shannon Entropy Situation:\n
    **Amplitude level**: {voltage_entropy_amplitude}\n
    **fluctuation level**: {voltage_entropy_fluctuation}\n
    **mean and std**: {voltage_entropy_mean_std}\n
    
    1.2. Temperature Characteristics:\n
    1.2.1. Average Voltage of the LIB Module:\n
    ......
                     
    1.3. SOC Characteristics:\n
    1.3.1. Average SOC of the LIB Module:\n
    ......
    (the template of temperature and soc is the the same as that of voltage)
    
    1.4. Current Characteristics:\n
    **Trend**: {current_trend}\n
    **Transition Events**: {current_transition_events}\n
    **Fluctuation**: {current_fluctuation}\n
    **Outlier Phenomenon**: {current_outlier_phenomenon}\n
    **mean and std (for stable phase)**: {current_mean_std}\n
    **initial/final values (for increasing or decreasing phase)**: {current_initial_final}\n
                     
     Please extract the data of LIB module based on the provided information and return it in the form of a JSON array. 
    All the elements constitute a JSON object containing the following keys:
    - "Overall_operation": {"overall_operation": "...", "overall_inconsistency": "..."}
    - "Voltage": {"average": { "trend": "...", "transitions": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}, 
                "maximum": { "trend": "...", "transitions": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
                "minimum": { "trend": "...", "transitions": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
                "standard_deviation": { "amplitude": "...", "fluctuation": "...", "mean_and_std": "..."}, 
                "shannon_entropy": { "amplitude": "...", "fluctuation": "...", "mean_and_std": "..."}}
    - "Temperature": {"average": { "trend": "...", "transitions": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}, 
                "maximum": { "trend": "...", "transitions": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
                "minimum": { "trend": "...", "transitions": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
                "standard_deviation": { "amplitude": "...", "fluctuation": "...", "mean_and_std": "..."}, 
                "shannon_entropy": { "amplitude": "...", "fluctuation": "...", "mean_and_std": "..."}}
    - "SOC": {"average": { "trend": "...", "transitions": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}, 
                "maximum": { "trend": "...", "transitions": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
                "minimum": { "trend": "...", "transitions": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
                "standard_deviation": { "amplitude": "...", "fluctuation": "...", "mean_and_std": "..."}, 
                "shannon_entropy": { "amplitude": "...", "fluctuation": "...", "mean_and_std": "..."}}
    - "Current": { "trend": "...", "transition_events": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}

    """)
    
    # Prompts for Cells 1~4
    SYSTEM_PROMPT2 = ("""
    You are an expert specializing in time-series signal analysis and maintenance for lithium-ion batteries.
    Your task is to complete the following template based on the specific information provided by the user.
    Attention: 
    1. Please extract the data of each LIB Cell (from Cell #1 to Cell #4) based on the provided information and return it in the form of a JSON array.
    2. The description should be in concise language. For instance:
    **in "trend", please output "increase (1~50th time), stable (50~100th time)" rather than "increase from 29.0°C to 30.0°C from 1 to 50th sample, stable from 30.0°C to 31.0°C from 51 to 100th sample."
    **in "mean and std", only the mean and std regarding "stable" phase are considered, please output like "stable at 3.2V, std is 0.0043 (1~50th time)."
    **in "initial/final values", ONLY initial/final values of "increase" (or "descrease") phases are considered, please output like "3.2V to 3.3V (1~50th time)."
    **in "fluctuation", please output "none" if user's information did not provide specific fluctuation level. Otherwise, please output like "fluctuations detected in 1~20th, 40~50th time." based on the user's information.
                      
    Instruct Example: when filling in the part of {cell_1_voltage_trend}, you will find "increase from 3.2V to 4.1V" in the user's information.
    then you can fill in {cell_1_voltage_trend} with "increase from 3.2V to 4.1V from 1 to 50th sample."
                                          """)
    
    text_template2 = ("""
    Output Template Example:
    LIB Cell #1:\n
    **Voltage**:\n
    trend: {cell_1_voltage_trend}\n
    transition events: {cell_1_voltage_transition_events}\n
    fluctuation: {cell_1_voltage_fluctuation}\n
    outliers: {cell_1_voltage_outliers}\n
    mean and std (for stable phase): {cell_1_voltage_mean_std}\n
    initial/final values (for increasing or decreasing phase): {cell_1_voltage_initial_final}\n

    **Temperature**:\n
    ......
                      
    **SOC**:\n
    ......
    (the template of temperature and soc is the the same as that of voltage)

    Please extract the data of each LIB Cell (from Cell #1 to Cell #4) based on the provided information and return it in the form of a JSON array:
    Each array element is a JSON object for each cell, containing the following keys:
    {"cell id": "...",
    "voltage": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "temperature": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "soc": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}}
    After generate the JSON object for each cell, please arrange them into a JSON array and return in the form as:
    {"cells_info": [{JSON for cell 1}, {JSON for cell 2}, ..., {JSON for cell 4}]} (please check the validity of the format!)   
    """)
    
    
    # Prompts for Cells 5~8
    SYSTEM_PROMPT3 = ("""
    You are an expert specializing in time-series signal analysis and maintenance for lithium-ion batteries.
    Your task is to complete the following template based on the specific information provided by the user.
    Attention: 
    1. Please extract the data of each LIB Cell (from Cell #5 to Cell #8) based on the provided information and return it in the form of a JSON array.
    2. The description should be in concise language. For instance:
    **in "trend", please output "increase (1~50th time), stable (50~100th time)" rather than "increase from 29.0°C to 30.0°C from 1 to 50th sample, stable from 30.0°C to 31.0°C from 51 to 100th sample."
    **in "mean and std", only the mean and std regarding "stable" phase are considered, please output like "stable at 3.2V, std is 0.0043 (1~50th time)."
    **in "initial/final values", ONLY initial/final values of "increase" (or "descrease") phases are considered, please output like "3.2V to 3.3V (1~50th time)."
    **in "fluctuation", please output "none" if user's information did not provide specific fluctuation level. Otherwise, please output like "fluctuations detected in 1~20th, 40~50th time." based on the user's information.
                      
    Instruct Example: when filling in the part of {cell_5_voltage_trend}, you will find "increase from 3.2V to 4.1V" in the user's information.
    then you can fill in {cell_5_voltage_trend} with "increase from 3.2V to 4.1V from 1 to 50th sample."
                                          """)
                                          
    text_template3 = ("""
    Output Template Example:
    LIB Cell #5:\n
    **Voltage**:\n
    trend: {cell_5_voltage_trend}\n
    transition events: {cell_5_voltage_transition_events}\n
    fluctuation: {cell_5_voltage_fluctuation}\n
    outliers: {cell_5_voltage_outliers}\n
    mean and std (for stable phase): {cell_5_voltage_mean_std}\n
    initial/final values (for increasing or decreasing phase): {cell_5_voltage_initial_final}\n

    **Temperature**:\n
    ......
                      
    **SOC**:\n
    ......
    (the template of temperature and soc is the the same as that of voltage)

    Please extract the data of each LIB Cell (from Cell #5 to Cell #8) based on the provided information and return it in the form of a JSON array:
    Each array element is a JSON object for each cell, containing the following keys:
    {"cell id": "...",
    "voltage": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "temperature": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "soc": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}}
    After generate the JSON object for each cell, please arrange them into a JSON array and return in the form as:
    {"cells_info": [{JSON for cell 5}, {JSON for cell 6}, ..., {JSON for cell 8}]} (please check the validity of the format!)
    """)
    
    # Prompts for Cells 9~12
    SYSTEM_PROMPT4 = ("""
    You are an expert specializing in time-series signal analysis and maintenance for lithium-ion batteries.
    Your task is to complete the following template based on the specific information provided by the user.
    Attention: 
    1. Please extract the data of each LIB Cell (from Cell #9 to Cell #12) based on the provided information and return it in the form of a JSON array.
    2. The description should be in concise language. For instance:
    **in "trend", please output "increase (1~50th time), stable (50~100th time)" rather than "increase from 29.0°C to 30.0°C from 1 to 50th sample, stable from 30.0°C to 31.0°C from 51 to 100th sample."
    **in "mean and std", only the mean and std regarding "stable" phase are considered, please output like "stable at 3.2V, std is 0.0043 (1~50th time)."
    **in "initial/final values", ONLY initial/final values of "increase" (or "descrease") phases are considered, please output like "3.2V to 3.3V (1~50th time)."
    **in "fluctuation", please output "none" if user's information did not provide specific fluctuation level. Otherwise, please output like "fluctuations detected in 1~20th, 40~50th time." based on the user's information.
                      
    Instruct Example: when filling in the part of {cell_9_voltage_trend}, you will find "increase from 3.2V to 4.1V" in the user's information.
    then you can fill in {cell_9_voltage_trend} with "increase from 3.2V to 4.1V from 1 to 50th sample."
                                          """)
    
    text_template4 = ("""
    Output Template Example:
    LIB Cell #9:\n
    **Voltage**:\n
    trend: {cell_9_voltage_trend}\n
    transition events: {cell_9_voltage_transition_events}\n
    fluctuation: {cell_9_voltage_fluctuation}\n
    outliers: {cell_9_voltage_outliers}\n
    mean and std (for stable phase): {cell_9_voltage_mean_std}\n
    initial/final values (for increasing or decreasing phase): {cell_9_voltage_initial_final}\n

    **Temperature**:\n
    ......
                      
    **SOC**:\n
    ......
    (the template of temperature and soc is the the same as that of voltage)

    Please extract the data of each LIB Cell (from Cell #9 to Cell #12) based on the provided information and return it in the form of a JSON array:
    Each array element is a JSON object for each cell, containing the following keys:
    {"cell id": "...",
    "voltage": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "temperature": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "soc": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}}
    After generate the JSON object for each cell, please arrange them into a JSON array and return in the form as:
    {"cells_info": [{JSON for cell 9}, {JSON for cell 10}, ..., {JSON for cell 12}]} (please check the validity of the format!)
    """)
    
    # Prompts for Cells 13~16
    SYSTEM_PROMPT5 = ("""
    You are an expert specializing in time-series signal analysis and maintenance for lithium-ion batteries.
    Your task is to complete the following template based on the specific information provided by the user.
    Attention: 
    1. Please extract the data of each LIB Cell (from Cell #13 to Cell #16) based on the provided information and return it in the form of a JSON array.
    2. The description should be in concise language. For instance:
    **in "trend", please output "increase (1~50th time), stable (50~100th time)" rather than "increase from 29.0°C to 30.0°C from 1 to 50th sample, stable from 30.0°C to 31.0°C from 51 to 100th sample."
    **in "mean and std", only the mean and std regarding "stable" phase are considered, please output like "stable at 3.2V, std is 0.0043 (1~50th time)."
    **in "initial/final values", ONLY initial/final values of "increase" (or "descrease") phases are considered, please output like "3.2V to 3.3V (1~50th time)."
    **in "fluctuation", please output "none" if user's information did not provide specific fluctuation level. Otherwise, please output like "fluctuations detected in 1~20th, 40~50th time." based on the user's information.
                      
    Instruct Example: when filling in the part of {cell_13_voltage_trend}, you will find "increase from 3.2V to 4.1V" in the user's information.
    then you can fill in {cell_13_voltage_trend} with "increase from 3.2V to 4.1V from 1 to 50th sample."
                                          """)
    
    text_template5 = ("""
    Output Template Example:
    LIB Cell #13:\n
    **Voltage**:\n
    trend: {cell_13_voltage_trend}\n
    transition events: {cell_13_voltage_transition_events}\n
    fluctuation: {cell_13_voltage_fluctuation}\n
    outliers: {cell_13_voltage_outliers}\n
    mean and std (for stable phase): {cell_13_voltage_mean_std}\n
    initial/final values (for increasing or decreasing phase): {cell_13_voltage_initial_final}\n

    **Temperature**:\n
    ......
                      
    **SOC**:\n
    ......
    (the template of temperature and soc is the same as that of voltage)

    Please extract the data of each LIB Cell (from Cell #13 to Cell #16) based on the provided information and return it in the form of a JSON array:
    Each array element is a JSON object for each cell, containing the following keys:
    {"cell id": "...",
    "voltage": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "temperature": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "soc": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}}
    After generate the JSON object for each cell, please arrange them into a JSON array and return in the form as:
    {"cells_info": [{JSON for cell 13}, {JSON for cell 14}, ..., {JSON for cell 16}]} (please check the validity of the format!)
    """)
    

    return [SYSTEM_PROMPT1 + clean_manual_linebreaks(text_template1), SYSTEM_PROMPT2 + clean_manual_linebreaks(text_template2),
    SYSTEM_PROMPT3 + clean_manual_linebreaks(text_template3),  SYSTEM_PROMPT4 + clean_manual_linebreaks(text_template4),
    SYSTEM_PROMPT5 + clean_manual_linebreaks(text_template5)]

def expert_prompt_for_report_mit():
    """Generate expert prompt templates for lithium-ion battery time-series analysis reports. (MIT)
    
    Constructs a set of standardized templates for describing battery cell characteristics,
    including voltage, temperature, capacity, and current parameters. The templates are designed to be
    filled with specific time-series analysis results.
    
    Returns:
        list: A list of 1 prompt strings, each combining a system prompt with a template
             
        Each prompt includes:
        - Detailed instructions for the analyst
        - Structured output templates with placeholders
        - JSON schema specifications for the expected output format
    """    
    
    text_template = ("""
    Output Template Example:
    LIB Cell #1:\n
    **Voltage**:\n
    trend: [cell_1_voltage_trend]\n
    transition events: [cell_1_voltage_transition_events]\n
    fluctuation: [cell_1_voltage_fluctuation]\n
    outliers: [cell_1_voltage_outliers]\n
    mean and std (for stable phase): [cell_1_voltage_mean_std]\n
    initial/final values (for increasing or decreasing phase): [cell_1_voltage_initial_final]\n

    **Temperature**:\n
    ......
                      
    **Charge capacity**:\n
    ......
    
    **Discharge capacity**:\n
    ......
    
    **Current**:\n
    ......
    
    (the template of temperature, charge capacity, discharge capacity, and current is the same as that of voltage)
    

    Please extract the data of each LIB Cell (from Cell #1 to Cell #6) based on the provided information and return it in the form of a JSON array:
    Each array element is a JSON object for each cell, containing the following keys:
    {"cell id": "...",
    "voltage": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "temperature": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "charge capacity": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "discharge capacity": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "current": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}}
    After generate the JSON object for each cell, please arrange them into a JSON array and return in the form as:   
    """
    '{"cells_info": [{JSON for cell 1}, {JSON for cell 2}, ..., {JSON for cell 6}]} (please check the validity of the format!)'
    )
    
    template_prepared = clean_manual_linebreaks(text_template)
   
    SYSTEM_PROMPT = ("""
    You are an expert specializing in time-series signal analysis and maintenance for lithium-ion batteries.
    Your task is to complete the following template based on the specific information provided by the user.
    Attention: 
    1. Please extract the data of each LIB Cell (from Cell #1 to Cell #6) based on the provided information and return it in the form of a JSON array.
    2. The description should be in concise language. For instance:
    **in "trend", please output "increase (1~50th time), stable (50~100th time)" rather than "increase from 29.0°C to 30.0°C from 1 to 50th sample, stable from 30.0°C to 31.0°C from 51 to 100th sample."
    **in "mean and std", only the mean and std regarding "stable" phase are considered, please output like "stable at 3.2V, std is 0.0043 (1~50th time)."
    **in "initial/final values", ONLY initial/final values of "increase" (or "descrease") phases are considered, please output like "3.2V to 3.3V (1~50th time)."
    **in "fluctuation", please output "none" if user's information did not provide specific fluctuation level. Otherwise, please output like "fluctuations detected in 1~20th, 40~50th time." based on the user's information.
                      
    Instruct Example: when filling in the part of [cell_1_voltage_trend], you will find "increase from 3.2V to 4.1V" in the user's information.
    then you can fill in [cell_1_voltage_trend] with "increase from 3.2V to 4.1V from 1 to 50th sample."
                                          """)
    
    return [SYSTEM_PROMPT, template_prepared]


def expert_prompt_for_report_tju():
    
    text_template = ("""
    Output Template Example:
    LIB Cell #1:\n
    **Voltage**:\n
    trend: [cell_1_voltage_trend]\n
    transition events: [cell_1_voltage_transition_events]\n
    fluctuation: [cell_1_voltage_fluctuation]\n
    outliers: [cell_1_voltage_outliers]\n
    mean and std (for stable phase): [cell_1_voltage_mean_std]\n
    initial/final values (for increasing or decreasing phase): [cell_1_voltage_initial_final]\n
                      
    **charge capacity**:\n
    ......
    
    **discharge capacity**:\n
    ......
    
    **current**:\n
    ......
    (the template of charge capacity, discharge capacity and current is the same as that of voltage.)

    Please extract the data of each LIB Cell (from Cell #1 to Cell #4) based on the provided information and return it in the form of a JSON array:
    Each array element is a JSON object for each cell, containing the following keys:
    {"cell id": "...",
    "voltage": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "charge capacity": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "discharge capacity": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."},
    "current": { "trend": "...", "transition": "...", "fluctuation": "...", "outliers": "...", "mean_and_std": "...", "initial_final": "..."}}
    After generate the JSON object for each cell, please arrange them into a JSON array and return in the form as:   
    {"cells_info": [{JSON for cell 1}, {JSON for cell 2}, ..., {JSON for cell 4}]} (please check the validity of the format!)"""
    )
    
    template_prepared = clean_manual_linebreaks(text_template)
    
    
    SYSTEM_PROMPT = ("""
    You are an expert specializing in time-series signal analysis and maintenance for lithium-ion batteries.
    Your task is to complete the following template based on the specific information provided by the user.
    Attention: 
    1. Please extract the data of each LIB Cell (from Cell #1 to Cell #4) based on the provided information and return it in the form of a JSON array.
    2. The description should be in concise language. For instance:
    **in "trend", please output "increase (1~50th time), stable (50~100th time)" rather than "increase from 29.0°C to 30.0°C from 1 to 50th sample, stable from 30.0°C to 31.0°C from 51 to 100th sample."
    **in "mean and std", only the mean and std regarding "stable" phase are considered, please output like "stable at 3.2V, std is 0.0043 (1~50th time)."
    **in "initial/final values", ONLY initial/final values of "increase" (or "descrease") phases are considered, please output like "3.2V to 3.3V (1~50th time)."
    **in "fluctuation", please output "none" if user's information did not provide specific fluctuation level. Otherwise, please output like "fluctuations detected in 1~20th, 40~50th time." based on the user's information.
                      
    Instruct Example: when filling in the part of [cell_1_voltage_trend], you will find "increase from 3.2V to 4.1V" in the user's information.
    then you can fill in [cell_1_voltage_trend] with "increase from 3.2V to 4.1V from 1 to 50th sample."
                                          """)
    
    return [SYSTEM_PROMPT, template_prepared]

def evaluate_prompt():
    """Generate an evaluation prompt for assessing LLM descriptions of battery time-series data.
    
    Constructs a detailed instruction set for scoring the factual accuracy of LLM response
    
    The prompt includes:
    - Clear scoring criteria (1.0, 0.5, 0.0 scale)
    - Step-by-step evaluation process
    - Concrete examples
    - Required output format
    - Important considerations for evaluation
    
    Returns:
        A formatted string containing the complete evaluation instructions
    """
    
    # Define the prompt components as a tuple for better readability
    prompt_components = (
        "I asked the LLM to describe a battery time series from a specific perspective, such as the trend, transition, etc. Now your task is to evaluate its description.",
        "Please score the following large language model (LLM) responses on a factual accuracy scale.",
        "Your scoring process should be as follows:\n1. Break down each response into multiple short sentences by splitting at punctuation marks.",
        "2. For each short sentence, assign a score based on its factual correctness according to its attached description of the facts:",
        "- **1.0** for a fact that is completely correct.\n- **0.5** for a fact that is partially correct. (not very precise but close. For instance, predicted transition point is 9th sample, while reality is 7th sample.)\n- **0.0** for a fact that is clearly incorrect.",
        "If the response contains only one short sentence, output that score directly, otherwise, calculate and output the average score of all sentences (keep 3 significant digits).",
        "Instruct example: ",
        """**LLM's Response:** trend: decrease (1~20th time), increase (21~40th time), stable (41~100th time).
        **Fact:** trend: decrease (1~22th time), increase (23~40th time), stable (41~100th time).
        **Scoring Breakdown:**
- **Clause 1:** "decrease (1~20th time)". As can be seen from the facts, trend of 1~20th time is 'decrease'. So it's correct. The score is 1.0.
- **Clause 2:** "increase (21~40th time)". As can be seen from the facts, trend of 21~22th time is 'decrease', while 23~40th time is 'increase'. So it is partially correct. The score is 0.5.
- **Clause 3:** "stable (41~100th time)". As can be seen from the facts, trend of 41~100th time is 'stable'. The score is 1.0.
**Final Score:** (1.0 + 0.5 + 1.0) / 3 = 0.8333""",
        
        "**Your final output must be a JSON array.** Each object in the array should contain the original `id` and the calculated `score` for that data entry.",
        """**Example of the expected JSON output: {"evaluate results": [{"id": 1,"score": 0.67},{"id": 2,"score": ...},...]}.""",
        "The list consisting of the responses of LLM to be evaluated and the corresponding facts is as follows.",
        "**(Please check carefully and ensure that all the LLM responses in the list got their corresponding scores!)",
        "Pay attention to certain words, such as the description of 'fluctuation', the meanings expressed by 'minimal', 'none', and 'slight' are actually the same, meaning that the fluctuation is negligible."
    )
    
    # Join components with newlines for the final prompt
    return "\n".join(prompt_components)