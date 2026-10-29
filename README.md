# Lopaca
Lopaca is a llama-server GUI frontend written in python using standard libraries, currently designed for Linux with GTK but others will come later. Primarily designed for local model usage with local documents. Features include RAG document support and rudimentary agentic support.

# Requirements
Should run using default packages on any recent(2026) Gnome-based distro, no extra packages needed.\
Python 3.14+ is required. For any recent Ubuntu-based distros WITHOUT gnome you should only need to run:
```
sudo apt install python3-gi gir1.2-adw-1
```
Edit settings.py if you need to change your server URL, default is currently latest llama-server default.

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
Installs to ~/.local/share/lopaca/ with a shortcut menu entry at ~/.local/share/applications/Lopaca.desktop.\
Then just run it using your menu shortcut.

# Local documents
Add readable text documents through a chat's menu, or place them in that chat's `documents/` folder. Before each prompt, a background worker checks for new, changed, and deleted files and updates `documents.sqlite` incrementally.

Retrieval uses SQLite FTS5/BM25, phrase and identifier matching, and filename/section searches combined with Reciprocal Rank Fusion (RRF). Relevant excerpts are injected into the current request as document reference data, not saved in chat history.

**Expand Document Searches** in App Settings optionally asks the same llama-server model for related search terms. It's off by default because it adds response latency.\
**Model-Assisted Document Indexing** in App settings optionally allows the LLM to help process documents for better info retrieval. Very slow and depends on your model speed. Modest quality improvement. Recommended for t/s greater than 80.

# TO-DO
- Add more global settings
- Add more individual chat settings
- Global memories file that the RAG function uses
- Selectable assistant style.md for communication styles
- More advanced agentic mode with more tools, web searching, etc

# Further TO-DOs
- Context compression using RAG once a limit is hit
- Make llama-server launchable within Lopaca with parameters
- Add support for other llm servers
- Add Image and Code tabs, image can connect to stable-diffusion.cpp

