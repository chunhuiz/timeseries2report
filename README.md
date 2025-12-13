<div align="center">

# TimeSeries2Report prompting enables adaptive large language model management of lithium-ion batteries

_This is the official implementation of [Self-Reflection Enhances Large Language Models Towards Substantial Academic Response](https://doi.org/10.1038/s44387-025-00045-3) in npj Artificial Intelligence, which can generate responses to academic paper review comments. The system supports using a reflection library to enhance response quality._

</div>


## Method
✅ **Transform time-series data into textual reports**

✅ **Enable LLM's application in time-series analysis without any training**

✅ **Showcase LLM-driven battery management--SOC predcition, anomaly detection, charging/discharging management**

We present TimeSeries2Report (TS2R), a prompting framework that converts raw lithium-ion battery operational time-series into structured, semantically enriched reports, enabling LLMs to reason, predict, and make decisions in BESS management scenarios. TS2R encodes short-term temporal dynamics into natural language through a combination of segmentation, semantic abstraction, and rule-based interpretation, effectively bridging low-level sensor signals with high-level contextual insights.

**TS2R-integrated LLMs achieve expert level decision quality and predictive consistency without retraining or architecture modification, establishing a practical path for adaptive, LLM-driven battery intelligence.**

![Dual-loop reflection method](figs/method_00.png)
<p align="center"><em>Figure: Dual-loop reflection method overview</em></p>

## Features

1. **PDF Paper Parsing**: Upload PDF files and automatically extract text content
2. **Review Comment Input**: Text input box for entering reviewer comments
3. **Reflection Mode**: Optional RAG reflection library enhancement feature that optimizes responses based on historical data
4. **Custom Rules Management**:
   - Save custom rules to local storage
   - Delete saved rules
   - Choose whether to insert custom rules into prompts
5. **API Configuration**: Configure model, API Key, and API URL
6. **Reasoning Content Display**: If the model supports reasoning, displays the reasoning process
7. **Two-Stage Response Generation**:
   - Initial Response: Generated directly based on the paper and reviewer comments
   - Final Response: Optimized initial response based on historical data from the reflection library

![Web application interface](figs/webpage_00.png)
<p align="center"><em>Figure: Web interface preview</em></p>

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Reflection Library

If you want to use reflection mode, ensure the following files exist:
- `./chroma_store_gpu/parent_child_store.json`
- `./chroma_store_gpu/chroma.sqlite3`
- `./all-MiniLM-L6-v2/` embedding model directory

### 3. Launch the Application

```bash
streamlit run app.py
```

The application will automatically open in your browser at: `http://localhost:8501`

## Windows Standalone (EXE)

No Python or extra packages required. To use the EXE:

1. Download the executable folder or zip.
2. Download and place the following two folders next to the EXE (same directory):
   - `all-MiniLM-L6-v2` (HuggingFace embedding model directory)
   - `chroma_store_gpu` (vector store directory with `chroma.sqlite3` and `parent_child_store.json`)

   Download links:
   - [all-MiniLM-L6-v2](https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2)
   - [chroma_store_gpu](https://drive.google.com/file/d/1_xj6eevkYw3nYbdGkWTeAOtxVqQHz2bQ/view?usp=sharing)

3. Expected directory layout:

```
PaperReviewLLM/
  ├── PaperReviewLLM_Dir.exe
  ├── _internal/
  ├── all-MiniLM-L6-v2/
  └── chroma_store_gpu/
      ├─ chroma.sqlite3
      └─ parent_child_store.json
```

4. Double-click `PaperReviewLLM_Dir.exe`. Your browser will open at `http://localhost:8501`.
5. To use Reflection Mode: enable it in the sidebar and click “Load Reflection Library”. The default path will automatically point to the `chroma_store_gpu` folder in the EXE directory.

Notes:
- The EXE bundles Python and required packages, but does not include the large `all-MiniLM-L6-v2` and `chroma_store_gpu` folders. Download and place them next to the EXE.
- The app looks for the `all-MiniLM-L6-v2` model folder next to the EXE.

## Quick Start

### Basic Mode (Without Reflection)

1. Configure Model, API Key, and API URL in the sidebar
2. Uncheck "Enable Reflection Mode"
3. Upload a PDF file on the left side of the main interface
4. Enter reviewer comments
5. Click "Generate Response"

### Using Reflection Mode

1. Configure Model, API Key, and API URL in the sidebar
2. Check "Enable Reflection Mode"
3. Enter the reflection library path (default: `./chroma_store_gpu`)
4. Click "Load Reflection Library" button
5. Upload a PDF file on the left side of the main interface
6. Enter reviewer comments
7. Click "Generate Response"


## Important Notes

1. **API Key Storage**: API keys are stored in memory during the session. You need to re-enter them after refreshing the page.

2. **Custom Rules Storage**: Rules are saved to the `custom_rule.txt` file and can be edited at any time. They remain accessible after restarting the application.

3. **Reflection Library Loading**: The reflection library only needs to be loaded once per session and remains available.

4. **Reasoning Content Processing**: If the model returns reasoning content (`reasoning_content`), it will be displayed in the initial response but will not be included in the messages during the reflection phase.

5. **Reflection Mode Requirements**: Using reflection mode requires loading the reflection library first. Ensure the vector library path is correct.

## Build (Developers)

Build a standalone EXE on Windows with PyInstaller:

1. Install dependencies (recommended in a clean venv):

```bash
pip install -r requirements.txt
pip install pyinstaller
```

2. Run the build command (bundles `app.py` and helper modules as data so Streamlit can run the script by path):

```bash
pyinstaller --noconfirm --clean --name "PaperReviewLLM_Dir" --onedir --windowed   --add-data "app.py;." --add-data "RAG_vectorstore.py;."   --collect-all "streamlit" --collect-all "langchain" --collect-all "chromadb"   --collect-data "langchain_chroma"   --collect-data "langchain_huggingface"   --hidden-import "pysqlite3"   launch.py
```

Notes:
- Entry point is `launch.py`, which programmatically runs `streamlit run app.py`.
- `--add-data` places `app.py` and `RAG_vectorstore.py` next to the unpacked EXE so Streamlit can run by file path.
- The large `all-MiniLM-L6-v2` and `chroma_store_gpu` folders are NOT bundled. Place them next to the EXE (see “Windows Standalone” section).
- The app reads/writes `custom_rule.txt` in the working directory (next to the EXE). Do NOT bundle it so users can edit and persist their rules.

## 🔗 Citation

If you find this code useful for your research, please cite

- [Baoxue Li and Chunhui Zhao, Self-Reflection Enhances Large Language Models Towards Substantial Academic Response. npj Artificial Intelligence, 2025, 1, Art. no. 42.](https://doi.org/10.1038/s44387-025-00045-3)

```BibTeX
@ARTICLE{10937933,
  author={Li, Baoxue and Zhao, Chunhui},
  journal={npj Artificial Intelligence}, 
  title={Self-Reflection Enhances Large Language Models Towards Substantial Academic Response}, 
  year={2025},
  volume={1},
  number={42},
  doi={10.1038/s44387-025-00045-3}}


```
