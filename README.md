# <img src="./lopaca.svg" width="50" height="50" align="center" alt="Lopaca Logo"> Lopaca
Lopaca is a llama-server GUI frontend written in python using standard libraries, currently designed for Linux with GTK but others will come later. Primarily designed for local model usage with local documents. Features include RAG document support and rudimentary agentic support.

# Requirements
Should run using default packages on any recent(2026) Gnome-based distro, no extra packages needed.\
Python 3.14+ is required. For any recent Ubuntu-based distros WITHOUT gnome you should only need to run:
```
sudo apt install python3-gi gir1.2-adw-1
```
Edit src/settings.py if you need to change your server URL, default is currently latest llama-server default.

# Usage
Start an instance of llama-server with your favorite LLM. Mostly tested with Qwen3.6 and 3.8 27b.\
To run:
```
./lopaca.sh
```
To install:
```
./install.sh
```
Or right click and run as program. May need to check Properties/Executable as Program first.\
Installs to ~/.local/share/lopaca/app/ with a shortcut menu entry at ~/.local/share/applications/app.lopaca.Lopaca.desktop.\
Then just run it using your menu shortcut.

To uninstall, run `./uninstall.sh`. You can keep your chats and settings or delete all Lopaca data.

# Local documents
Choose **Add Documents** from a chat’s menu and drag files or folders into the drop box. The folder button beside **Send** opens the same box and tells the model about the documents for your next message in that chat. Lopaca copies supported documents into that chat's `documents/` folder. You can also place files there directly. Before each prompt, a background worker checks for new, changed, and deleted files and updates `documents.sqlite` incrementally.

Retrieval uses SQLite FTS5/BM25, phrase and identifier matching, and filename/section searches combined with Reciprocal Rank Fusion (RRF). Relevant excerpts are injected into the current request as document reference data, not saved in chat history.

DOCX, PPTX, and XLSX documents require optional Python libraries `python-docx`, `python-pptx`, and `openpyxl`, respectively. OpenDocument text (`.odt`), presentations (`.odp`), and spreadsheets (`.ods`) require `odfpy`. EPUB books require `EbookLib`. Lopaca warns when a required library is missing. Install it in the Python environment that runs Lopaca, then restart Lopaca. Files already in `documents/` will be retried. These libraries are optional and are not installed by the installer.

RAG and file tools share the same text extraction. DOCX paragraphs and tables, PPTX slide text and speaker notes, and XLSX sheet names and cell coordinates are preserved in extracted text. Spreadsheet formulas are retained without calculation. File tools extract packaged documents on demand; indexed passages are cached in `documents.sqlite`. OpenDocument extraction preserves text and table rows, including slide/sheet names where supplied. EPUB chapters follow the book’s reading order. Standalone HTML and other text sources remain unmodified. Original files stay intact, and citation line numbers refer to extracted text.

**Expand Document Searches** in App Settings optionally asks the same llama-server model for related search terms. It's off by default because it adds response latency.

**Model-Assisted Document Indexing** in App settings optionally allows the LLM to help process documents for better info retrieval. Very slow and depends on your model speed. Modest quality improvement. Recommended for t/s greater than 80.

# TO-DO
- Add embedding model support?
- Add PDF and other docs support
- Make features list
- More advanced agentic mode with more tools, web searching, etc

# Further TO-DOs
- Add image sending to llm for vision equipped models(not image gen)
- Add support for other llm servers
- Add Image and Code tabs, image can connect to stable-diffusion.cpp
- Add cross-chat history RAG?


# Support Lopaca
[Buy me a coffee!](https://buymeacoffee.com/portets)