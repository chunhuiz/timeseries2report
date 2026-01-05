# app.py
from numba.core.types import none
import concurrent.futures
import streamlit as st
from typing import Optional
import time
import os
import json
import pickle
from dotenv import load_dotenv
from openai import OpenAI
import utils.data_utils as du
import utils.prompt_engineering as pe
from utils.db_utils import (
    init_db,
    load_sessions_to_memory,
    add_message,
    get_messages,
    clear_messages,
    delete_session,
    update_session_name,
    create_session,
    save_analysis_result,
    get_analysis_result,
    delete_analysis_result
)
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import warnings
warnings.filterwarnings("ignore")
# Load environment variables
load_dotenv()

def convert_numpy_to_json_serializable(obj):
    """
    Recursively convert numpy arrays to Python lists for JSON serialization
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def markdown_table_to_pdf(markdown_text, descript_c2_json):
    """
    Convert markdown table to PDF with proper text wrapping
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
    )

    # Add title
    title = Paragraph("Battery Data Analysis Results", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))

    # Parse JSON data
    if isinstance(descript_c2_json, str):
        data = json.loads(descript_c2_json)
    else:
        data = descript_c2_json

    cells_info = data.get('cells_info', [])

    # Define cell style for wrapping text
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontSize=7,
        leading=9,
        wordWrap='CJK',
        alignment=1  # Center alignment
    )

    for cell in cells_info:
        cell_id = cell.get('cell id', 'Unknown')

        # Add cell title
        cell_title = Paragraph(f"<b>{cell_id}</b>", styles['Heading2'])
        story.append(cell_title)
        story.append(Spacer(1, 0.1*inch))

        # Create table data with Paragraphs for text wrapping
        table_data = [
            [Paragraph('<b>Metric</b>', cell_style),
             Paragraph('<b>Trend</b>', cell_style),
             Paragraph('<b>Transition</b>', cell_style),
             Paragraph('<b>Fluctuation</b>', cell_style),
             Paragraph('<b>Outliers</b>', cell_style),
             Paragraph('<b>Mean & Std</b>', cell_style),
             Paragraph('<b>Initial-Final</b>', cell_style)]
        ]

        metrics = {
            'voltage': 'Voltage',
            'charge capacity': 'Charge Capacity',
            'discharge capacity': 'Discharge Capacity',
            'current': 'Current'
        }

        for metric_key, metric_name in metrics.items():
            metric_data = cell.get(metric_key, {})

            # Convert all values to strings and wrap in Paragraphs
            row = [
                Paragraph(f"<b>{metric_name}</b>", cell_style),
                Paragraph(str(metric_data.get('trend', '-'))[:100], cell_style),  # Limit length
                Paragraph(str(metric_data.get('transition', '-'))[:100], cell_style),
                Paragraph(str(metric_data.get('fluctuation', '-'))[:100], cell_style),
                Paragraph(str(metric_data.get('outliers', '-'))[:100], cell_style),
                Paragraph(str(metric_data.get('mean_and_std', '-'))[:100], cell_style),
                Paragraph(str(metric_data.get('initial_final', '-'))[:100], cell_style)
            ]
            table_data.append(row)

        # Create table with adjusted column widths to fit page
        # Total width = 7.5 inches (8.5 - 1 inch margins)
        t = Table(table_data, colWidths=[1.0*inch, 1.1*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.2*inch, 1.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))

        story.append(t)
        story.append(Spacer(1, 0.3*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

def format_cells_info_to_markdown(descript_c2_json):
    """
    Format descript c2 JSON data into Markdown table
    """
    try:
        # Parse JSON string
        if isinstance(descript_c2_json, str):
            data = json.loads(descript_c2_json)
        else:
            data = descript_c2_json

        # Get cells_info
        cells_info = data.get('cells_info', [])

        if not cells_info:
            return "⚠️ No battery data found"

        # Create Markdown table
        markdown_output = "### 📊 Battery Data Analysis Results\n\n"

        for cell in cells_info:
            cell_id = cell.get('cell id', 'Unknown')
            markdown_output += f"#### {cell_id}\n\n"

            # Create table
            markdown_output += "| Metric | Trend | Transition | Fluctuation | Outliers | Mean & Std | Initial-Final |\n"
            markdown_output += "|--------|-------|------------|-------------|----------|------------|---------------|\n"

            # Process each measurement metric
            metrics = {
                'voltage': 'Voltage',
                'charge capacity': 'Charge Capacity',
                'discharge capacity': 'Discharge Capacity',
                'current': 'Current'
            }

            for metric_key, metric_name in metrics.items():
                metric_data = cell.get(metric_key, {})

                trend = metric_data.get('trend', '-')
                transition = metric_data.get('transition', '-')
                fluctuation = metric_data.get('fluctuation', '-')
                outliers = metric_data.get('outliers', '-')
                mean_and_std = metric_data.get('mean_and_std', '-')
                initial_final = metric_data.get('initial_final', '-')

                markdown_output += f"| **{metric_name}** | {trend} | {transition} | {fluctuation} | {outliers} | {mean_and_std} | {initial_final} |\n"

            markdown_output += "\n"

        return markdown_output

    except Exception as e:
        return f"⚠️ Data parsing failed: {str(e)}"

def process_task(
    system_prompt: str,
    user_content: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> Optional[str]:
    """
    Thread task function: Process single API call, return JSON result (or None if failed).
    """
    max_retries = 3  # Maximum retry attempts
    retry_delay_base = 1.0  # Base delay seconds (exponential backoff)

    for attempt in range(max_retries + 1):
        try:
            # Construct conversation messages
            messages = [
                {"role": "system", "content": f"(IMPORTANT: This is an isolated request.){system_prompt}"},
                {"role": "user", "content": user_content}
            ]
            client = OpenAI(api_key=api_key, base_url=base_url)

            # Call API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                extra_body={"enable_thinking": False},
            )

            result = response.choices[0].message.content.strip()

            # Validate result (critical step)
            validate1, validate2 = du.validate_result(result)

            if validate1:
                if len(validate2) == 0:
                    return result
                else:
                    return validate2[0]
            else:
                print(f"Result validation failed (attempt {attempt+1}/{max_retries+1})")
                if attempt < max_retries:
                    print(f"Waiting {retry_delay_base * (2 ** attempt)} seconds before retry...")

        except Exception as e:
            print(f"API call error (attempt {attempt+1}/{max_retries+1}): {str(e)}")

        # Delay before retry (exponential backoff)
        if attempt < max_retries:
            delay = retry_delay_base * (2 ** attempt)
            time.sleep(delay)

    print(f"⚠️ Task finally failed")
    return None

def multivariate_seed_qa(model_config: dict, data_v_seen, data_c_seen, data_v_pre, data_c_pre, data_desp_i, char_state):
    SYSTEM_PROMPT2, template2 = pe.prompt_for_list()
    instruction = (f"There are time series collected from 4 LIB cells.")

    time_series_list = [
    data_v_seen[:, i, j]  
    for j in range(data_v_seen.shape[2])
    for i in range(data_v_seen.shape[1])]
    time_series_list.append(data_c_seen.flatten())

    SYSTEM_PROMPT_list = [SYSTEM_PROMPT2]
    template_list = [template2]
    des_pick = ["ind1"]
    des_json_list = []

    tasks = []
    for sys_prompt, template, des_key in zip(SYSTEM_PROMPT_list, template_list, des_pick):
        user_content = instruction + data_desp_i[des_key]  # Construct user content
        sys_prompt_combined = sys_prompt + template
        tasks.append((sys_prompt_combined, user_content))  # Each task contains two parameters

    # 3. Execute tasks in parallel using thread pool
    des_json_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit tasks: use lambda to pass fixed parameters (client, model, etc.), executor.map returns results in order
        results = executor.map(
            lambda task: process_task(
                task[0],  # system_prompt (from task list)
                task[1],  # user_content (from task list)
                model_config['api_key'],
                model_config['base_url'],
                model_config['model_name'],  # Model name (fixed)
                model_config['temperature'],   # Temperature (fixed)
                model_config['max_tokens']     # Max tokens (fixed)
            ),
            tasks  # Task list
        )

        # Collect results: filter out failed tasks (those returning None)
        des_json_list = [res for res in results if res is not None]

    # Check if there are valid results
    if len(des_json_list) == 0:
        raise ValueError("All API calls failed, unable to generate data description. Please check model configuration, API key and network connection.")

    result = {
        'original list': "\n".join(data_desp_i),
        'time series': time_series_list,
        'descript c2': des_json_list[0],
    }

    return result

def generate_seed_qa_dataset(model_config: dict, data_v_win_train, data_c_win_train):
    dir_path = os.getcwd()
    OUTPUT_DATASET = f'{dir_path}/template_nca_description.json'
    item = none
    with open(OUTPUT_DATASET, 'wt') as f:
        i = 0
        data_v_i = data_v_win_train[i]
        data_c_i = data_c_win_train[i]
        # Determine charging or discharging
        if i >= 20:
            char_state = 0
        else:
            char_state = 1
        data_desp_i = du.generate_descriptions(data_v_i, data_c_i, window=10, charge=char_state)
        
        item = multivariate_seed_qa(model_config, data_v_i, data_c_i, 0, 0, data_desp_i,char_state)
        
        cur_data = {
            'original list': item['original list'],
            'time series': du.timeseries_to_list(item['time series']),
            'descript c2': item['descript c2'],
        }
        f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')
    return item

def geenerate_data_prompt(uploaded_file, model_config):
    """Process uploaded file and generate data prompt"""
    dataset = none
    # Streamlit's UploadedFile object can be directly read as byte stream
    dataset = pickle.load(uploaded_file)
    data_v_win_train, data_c_win_train = du.read_data_from_nca(dataset)
    data_prompt = generate_seed_qa_dataset(model_config, data_v_win_train, data_c_win_train)
    return data_prompt

def main_interface():
    """Main interface function, supports multiple historical conversation windows"""
    # Set page configuration
    st.set_page_config(page_title="Battery LLM Assistant", page_icon="🔋", layout="wide")

    # Initialize database
    init_db()

    # Initialize session list (load from database)
    if "sessions" not in st.session_state:
        st.session_state.sessions = load_sessions_to_memory()
        st.session_state.current_session = list(st.session_state.sessions.keys())[0]
        st.session_state.db_synced = True

    # Initialize data file upload status for each session
    if "session_data_uploaded" not in st.session_state:
        st.session_state.session_data_uploaded = {}

    # Initialize dataset for each session
    if "session_datasets" not in st.session_state:
        st.session_state.session_datasets = {}

    # Initialize model selection for each session
    if "session_models" not in st.session_state:
        st.session_state.session_models = {}

    # Initialize analysis table for each session
    if "session_analysis_tables" not in st.session_state:
        st.session_state.session_analysis_tables = {}

    # Load analysis results from database for all sessions
    if "analysis_loaded" not in st.session_state:
        for session_name in st.session_state.sessions.keys():
            saved_result = get_analysis_result(session_name)
            if saved_result:
                # Restore dataset
                st.session_state.session_datasets[session_name] = saved_result['result_data']
                # Restore markdown table
                st.session_state.session_analysis_tables[session_name] = saved_result['markdown_table']
                # Mark as data uploaded
                st.session_state.session_data_uploaded[session_name] = True

                # Add system message to conversation context if not already exists
                markdown_result = saved_result['markdown_table']
                current_messages = st.session_state.sessions[session_name]
                has_system_message = any(msg.get("role") == "system" for msg in current_messages)

                if not has_system_message:
                    system_context_message = {
                        "role": "system",
                        "content": f"The following are the battery data analysis results for this conversation. Users may ask questions based on these data:\n\n{markdown_result}\n\nPlease answer user questions based on these data."
                    }
                    current_messages.insert(0, system_context_message)

        st.session_state.analysis_loaded = True

    # Ensure current session has upload status record
    if st.session_state.current_session not in st.session_state.session_data_uploaded:
        st.session_state.session_data_uploaded[st.session_state.current_session] = False

    # Ensure current session has model selection record (default to model from environment variable)
    if st.session_state.current_session not in st.session_state.session_models:
        default_model = os.getenv("MODEL_NAME", "qwen3-14b")
        st.session_state.session_models[st.session_state.current_session] = default_model

    # Sidebar - Session management
    with st.sidebar:
        st.header("📝 Conversation Windows")

        # New session button
        if st.button("➕ New Session", use_container_width=True):
            # Find the next available session number
            existing_numbers = []
            for name in st.session_state.sessions.keys():
                # Extract number from "Session X" format
                if name.startswith("Session "):
                    try:
                        num = int(name.split(" ")[1])
                        existing_numbers.append(num)
                    except:
                        pass

            # Find next available number
            if existing_numbers:
                next_num = max(existing_numbers) + 1
            else:
                next_num = 1

            new_session_name = f"Session {next_num}"
            st.session_state.sessions[new_session_name] = []
            st.session_state.current_session = new_session_name
            # Mark new session as no data uploaded
            st.session_state.session_data_uploaded[new_session_name] = False
            # Sync to database
            create_session(new_session_name)
            st.rerun()

        st.divider()

        # Initialize editing state
        if "editing_session" not in st.session_state:
            st.session_state.editing_session = None

        # List all sessions
        for session_name in list(st.session_state.sessions.keys()):
            is_current = session_name == st.session_state.current_session
            is_editing = st.session_state.editing_session == session_name

            # If editing this session name
            if is_editing:
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    new_name = st.text_input(
                        "Rename",
                        value=session_name,
                        key=f"rename_{session_name}",
                        label_visibility="collapsed"
                    )
                    if new_name != session_name and new_name:
                        # Rename session
                        st.session_state.sessions[new_name] = st.session_state.sessions.pop(session_name)
                        if st.session_state.current_session == session_name:
                            st.session_state.current_session = new_name
                        st.session_state.editing_session = None
                        # Sync to database
                        update_session_name(session_name, new_name)
                        st.rerun()
                with col2:
                    if st.button("✓", key=f"confirm_{session_name}", use_container_width=True):
                        st.session_state.editing_session = None
                        st.rerun()
            else:
                # Display session name and action buttons
                col1, col2, col3 = st.columns([0.7, 0.15, 0.15])

                with col1:
                    button_type = "primary" if is_current else "secondary"
                    if st.button(
                        f"{'📌 ' if is_current else ''}{session_name}",
                        key=f"select_{session_name}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state.current_session = session_name
                        st.rerun()

                with col2:
                    if st.button("✏️", key=f"edit_{session_name}", use_container_width=True):
                        st.session_state.editing_session = session_name
                        st.rerun()

                with col3:
                    # Delete button (keep at least one session)
                    if len(st.session_state.sessions) > 1:
                        if st.button("🗑️", key=f"delete_{session_name}", use_container_width=True):
                            del st.session_state.sessions[session_name]
                            # Delete corresponding data upload status
                            if session_name in st.session_state.session_data_uploaded:
                                del st.session_state.session_data_uploaded[session_name]
                            # Delete corresponding dataset
                            if session_name in st.session_state.session_datasets:
                                del st.session_state.session_datasets[session_name]
                            # Delete corresponding analysis table
                            if session_name in st.session_state.session_analysis_tables:
                                del st.session_state.session_analysis_tables[session_name]
                            if st.session_state.current_session == session_name:
                                st.session_state.current_session = list(st.session_state.sessions.keys())[0]
                            # Sync to database (delete session and analysis results)
                            delete_session(session_name)
                            delete_analysis_result(session_name)
                            st.rerun()

        st.divider()

        # Clear current session history
        if st.button("🧹 Clear Current Conversation", use_container_width=True):
            st.session_state.sessions[st.session_state.current_session] = []
            # Sync to database
            clear_messages(st.session_state.current_session)

            # If there's analysis result, re-add system message
            if st.session_state.current_session in st.session_state.session_analysis_tables:
                markdown_result = st.session_state.session_analysis_tables[st.session_state.current_session]
                system_context_message = {
                    "role": "system",
                    "content": f"The following are the battery data analysis results for this conversation. Users may ask questions based on these data:\n\n{markdown_result}\n\nPlease answer user questions based on these data."
                }
                st.session_state.sessions[st.session_state.current_session].append(system_context_message)
                # Also add to database
                add_message(st.session_state.current_session, "system", system_context_message["content"])

            st.rerun()

    # Main interface title
    st.title(f"💬 {st.session_state.current_session}")

    # File upload and model selection area - side by side
    col_left_space, col_upload, col_model, col_right_space = st.columns([0.5, 2, 2, 0.5])

    # Model selection area
    with col_model:
        st.markdown("##### 🤖 Model Selection")
        available_models = ['qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b-instruct-2507', 'qwen3-max']
        current_model = st.session_state.session_models.get(st.session_state.current_session, "qwen3-14b")

        # Get current model index in the list
        try:
            current_index = available_models.index(current_model)
        except ValueError:
            current_index = 0

        selected_model = st.selectbox(
            "Select conversation model",
            options=available_models,
            index=current_index,
            key=f"model_selector_{st.session_state.current_session}"
        )

        # Update current session's model selection
        st.session_state.session_models[st.session_state.current_session] = selected_model

        # Display model information
        st.caption(f"Current model: **{selected_model}**")

    st.divider()

    # Get environment variables
    api_key = os.getenv("CHAT_API_KEY")
    base_url = os.getenv("BASE_URL")
    # Use model selected for current session
    model_name = st.session_state.session_models.get(st.session_state.current_session, "qwen3-14b")
    model_config = {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
        "temperature": float(os.getenv("TEMPERATURE", "0.3")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "8192"))
    }

    # File upload area
    with col_upload:
        st.markdown("##### 📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload data file (Pickle)",
            type=["pkl", "pickle"],
            key=f"file_uploader_{st.session_state.current_session}"
        )

        if uploaded_file is not None:
            st.info(f"📄 {uploaded_file.name} ({uploaded_file.size} bytes)")
            if st.button("Process File", use_container_width=True, type="primary", key=f"process_file_{st.session_state.current_session}"):
                with st.spinner("Processing file..."):
                    try:
                        # Call file processing function (user-implemented)
                        result = geenerate_data_prompt(uploaded_file, model_config)

                        # Convert numpy arrays to lists for JSON serialization
                        result_serializable = convert_numpy_to_json_serializable(result)

                        # Save dataset to current session (use serializable version)
                        st.session_state.session_datasets[st.session_state.current_session] = result_serializable
                        # Mark current session as data uploaded
                        st.session_state.session_data_uploaded[st.session_state.current_session] = True

                        # Display formatted analysis results
                        if result_serializable and 'descript c2' in result_serializable:
                            markdown_result = format_cells_info_to_markdown(result_serializable['descript c2'])

                            # Save analysis table to current session
                            st.session_state.session_analysis_tables[st.session_state.current_session] = markdown_result

                            # Save analysis result to database for persistence
                            save_analysis_result(
                                st.session_state.current_session,
                                result_serializable,
                                markdown_result
                            )

                            st.markdown(markdown_result)

                            # Add analysis results as system message to conversation context
                            system_context_message = {
                                "role": "system",
                                "content": f"The following are the battery data analysis results for this conversation. Users may ask questions based on these data:\n\n{markdown_result}\n\nPlease answer user questions based on these data."
                            }

                            # Add to current session's message list beginning (if not already added)
                            current_messages = st.session_state.sessions[st.session_state.current_session]
                            # Check if system message already exists
                            has_system_message = any(msg.get("role") == "system" for msg in current_messages)
                            if not has_system_message:
                                current_messages.insert(0, system_context_message)
                                # Also save system message to database
                                add_message(st.session_state.current_session, "system", system_context_message["content"])

                        st.success("File processing complete! Data loaded to conversation context, you can now ask questions based on analysis results.")
                        st.rerun()  # Rerun to display table
                    except Exception as e:
                        st.error(f"File processing failed: {str(e)}")

    # Check if current session has uploaded data (needed for visualization)
    data_uploaded = st.session_state.session_data_uploaded.get(st.session_state.current_session, False)

    st.divider()

    # Visualization area - only show when data is uploaded
    if data_uploaded and st.session_state.current_session in st.session_state.session_datasets:
        dataset = st.session_state.session_datasets[st.session_state.current_session]

        if 'time series' in dataset:
            st.markdown("##### 📈 Time Series Visualization")

            # Create two columns for selectors
            col_vis1, col_vis2, col_vis3 = st.columns([2, 2, 2])

            with col_vis1:
                # Cell selector (1-4)
                selected_cell = st.selectbox(
                    "Select Cell",
                    options=[1, 2, 3, 4],
                    key=f"cell_selector_{st.session_state.current_session}"
                )

            with col_vis2:
                # Metric selector
                metric_options = {
                    "Voltage": 0,
                    "Charge Capacity": 1,
                    "Discharge Capacity": 2,
                    "Current": 3
                }
                selected_metric = st.selectbox(
                    "Select Metric",
                    options=list(metric_options.keys()),
                    key=f"metric_selector_{st.session_state.current_session}"
                )

            # Calculate the index in time series list
            # time_series_list layout: cell1_voltage, cell1_charge, cell1_discharge, cell1_current, cell2_voltage, ...
            time_series_list = dataset['time series']
            series_index = (selected_cell - 1) * 4 + metric_options[selected_metric]

            if series_index < len(time_series_list):
                # Get the selected time series data
                selected_data = time_series_list[series_index]

                # Convert to numpy array if it's a list
                if isinstance(selected_data, list):
                    selected_data = np.array(selected_data)

                # Create the plot
                fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
                ax.plot(selected_data, linewidth=2, color='#1f77b4')
                ax.set_xlabel('Time Step', fontsize=12)
                ax.set_ylabel(selected_metric, fontsize=12)
                ax.set_title(f'Cell {selected_cell} - {selected_metric}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Display the plot
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

            st.divider()

    # If no data uploaded, show prompt
    if not data_uploaded:
        st.warning("⚠️ Please upload battery data file first before starting conversation analysis.")
    else:
        # If data uploaded, display analysis table at the top of conversation interface
        if st.session_state.current_session in st.session_state.session_analysis_tables:
            st.markdown(st.session_state.session_analysis_tables[st.session_state.current_session])

            # Add download buttons
            col_download1, col_download2, col_spacer = st.columns([1, 1, 3])

            with col_download1:
                # PDF download button
                if st.session_state.current_session in st.session_state.session_datasets:
                    dataset = st.session_state.session_datasets[st.session_state.current_session]
                    if 'descript c2' in dataset:
                        pdf_buffer = markdown_table_to_pdf(
                            st.session_state.session_analysis_tables[st.session_state.current_session],
                            dataset['descript c2']
                        )
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_buffer,
                            file_name="battery_analysis.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

            with col_download2:
                # JSON download button - download complete result
                if st.session_state.current_session in st.session_state.session_datasets:
                    dataset = st.session_state.session_datasets[st.session_state.current_session]
                    # Download the complete result including time series
                    json_str = json.dumps(dataset, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name="battery_data_complete.json",
                        mime="application/json",
                        use_container_width=True
                    )

            st.divider()

    # Get current session messages
    current_messages = st.session_state.sessions[st.session_state.current_session]

    # Display historical messages (skip system messages as they were shown during file processing)
    for message in current_messages:
        if message["role"] != "system":  # Don't display system messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input - only enabled after data upload
    chat_input_disabled = not data_uploaded
    prompt_placeholder = "Please upload data file first..." if chat_input_disabled else "Enter your question..."

    if prompt := st.chat_input(prompt_placeholder, disabled=chat_input_disabled):
        full_response = ""
        # Add user message
        current_messages.append({"role": "user", "content": prompt})
        # Sync to database
        add_message(st.session_state.current_session, "user", prompt)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                # Initialize client
                client = OpenAI(api_key=api_key, base_url=base_url)

                # Call LLM API
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in current_messages
                    ],
                    stream=True,
                )

                # Stream display response
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"Error occurred: {str(e)}"
                message_placeholder.error(full_response)

        # Add assistant response (outside chat_message block)
        current_messages.append({"role": "assistant", "content": full_response})
        # Sync to database
        add_message(st.session_state.current_session, "assistant", full_response)


if __name__ == "__main__":
    # Start main interface
    main_interface()
