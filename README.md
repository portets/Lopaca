# <img src="./lopaca.svg" width="50" height="50" align="center" alt="Lopaca Logo"> Lopaca
Lopaca is a llama-server GUI frontend written in python using standard libraries, currently designed for Linux with GTK but others will come later. Primarily designed for local model usage with local documents. Features include RAG document support and rudimentary agentic support.

# Requirements
Should run out of the box on any recent(2025+) Gnome-based distro, no extra packages needed.\
Python 3.13+ is required. For any recent Debian-based distros WITHOUT gnome you should only need to run:
```
sudo apt install python3-gi gir1.2-adw-1
```
Edit src/settings.py if you need to change your server URL, default is currently latest llama-server default.

# Usage
Start an instance of llama-server with your favorite LLM. Mostly tested with Qwen3.5 9b, 3.6 27b and 3.8 27b.\
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

Retrieval uses SQLite FTS5/BM25, phrase and identifier matching, and filename/section searches combined with Reciprocal Rank Fusion (RRF). With an embedding model loaded(automatic default) vectorization is additionally used with RRF. Relevant excerpts are injected into the current request as document reference data.

**The two below features are now replaced by an embedding model by defautl**

**Expand Document Searches** in App Settings optionally asks the same llama-server model for related search terms. It's off by default because it adds response latency.

**Model-Assisted Document Indexing** in App settings optionally allows the LLM to help process documents for better info retrieval. Very slow and depends on your model speed. Modest quality improvement. Recommended for t/s greater than 80.

# TO-DO
- Make features list
- Add Qwen Flash Next when llama-server gains moe-stream support

# Further TO-DOs
- More advanced agentic mode with more tools, web searching, etc
- Add support for other llm servers
- Windows and Mac support
- Add Image and Code tabs, image can connect to stable-diffusion.cpp


# Support Lopaca
[Buy me a coffee!](https://buymeacoffee.com/portets)
