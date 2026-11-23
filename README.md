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

DOCX, PPTX, and XLSX documents require optional Python libraries `python-docx`, `python-pptx`, and `openpyxl`, respectively. OpenDocument text (`.odt`), presentations (`.odp`), and spreadsheets (`.ods`) require `odfpy`. EPUB books require `EbookLib`, RTF documents require `striprtf`, and PDFs require `pypdf`. EML messages and MBOX mailboxes use Python’s standard library. To set up these libraries and the optional archive readers, choose **Download Optional Features** in General Settings or keep the optional features checkbox selected during model setup. Lopaca reuses available system libraries and installs missing ones into its own virtual environment. Restart Lopaca to use the environment. Files already in `documents/` will be retried.

Optional features also include high-performance Xet downloads for large Hugging Face files and OCRmyPDF for scanned PDFs. Lopaca uses its adaptive HTTP downloader when Xet is unavailable or when resuming an HTTP partial. Install Tesseract (`tesseract-ocr` on Debian and Ubuntu) to make scanned pages searchable. OCR runs during background indexing and preserves the original PDF.

NumPy speeds up vector searches across knowledge packs, documents, memories, and chat histories when an embedding model is in use. Lopaca uses the same saved vectors and can search them with standard Python when NumPy is unavailable.

**Expand Document Searches** in App Settings optionally asks the same llama-server model for related search terms. It's off by default because it adds response latency.

**Model-Assisted Document Indexing** in App settings optionally allows the LLM to help process documents for better info retrieval. Very slow and depends on your model speed. Modest quality improvement. Recommended for t/s greater than 80.

# TO-DO
- Add embedding model support?
- Make features list

# Further TO-DOs
- More advanced agentic mode with more tools, web searching, etc
- Add support for other llm servers
- Windows and Mac support
- Add Image and Code tabs, image can connect to stable-diffusion.cpp


# Support Lopaca
[Buy me a coffee!](https://buymeacoffee.com/portets)
