# Lopaca
Lopaca is a llama-server GUI frontend written in python using standard libraries, currently deisgned for Linux with GTK but others will come later. Primarily designed for local model usage with local documents. Features include RAG document support and rudimentary agentic support.

# Requirements
Should run using default packages on any recent(2026) Gnome-based distro, no extra packages needed.
Python 3.14+ is required. For any recent Ubuntu-based distros WITHOUT gnome you should only need to run:
```
sudo apt install python3-gi gir1.2-adw-1
```

# Usage
Start an instance of llama-server with your favorite LLM. Mostly tested with Qwen3.6 and 3.8 27b.
Make sure gtk python libraries are installed. Otherwise it's all standard libraries.
```
./lopaca.sh
```
Edit settings.py if you need top change your server URL, default is currently latest llama-server default.

# TO-DO
- Add global settings menu and populate
- Add more individual chat settings
- More robust RAG; BM25 and RFF
- Global memories file that the RAG function uses
- Selectable assistant style.md for communication styles
- Basic agentic mode

# Further TO-DOs
- Context compression using RAG once a limit is hit
- Make llama-server launchable within Lopaca with parameters
- Add support for other llm servers
- More advanced agentic mode with more tools, web searching, etc
- Add Image and Code tabs, image can connect to stable-diffusion.cpp

