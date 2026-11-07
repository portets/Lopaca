# Lopaca architecture

Lopaca is a local chat application for Python 3.14+ and GTK4/libadwaita. Runtime dependencies
outside the standard library are limited to the GTK-related bindings and libraries. Model
inference runs in an external llama-server process.

## Module responsibilities

| Module | Owns |
| --- | --- |
| `src/main.py` | Entry point, `ChatController`, application state, chat/settings actions, request preparation, worker coordination and UI notifications. |
| `src/ui.py` | GTK application/window, CSS, sidebar, transcript layout, composer, settings/memory/approval dialogs, status timers and stream rendering updates. |
| `src/markdown_ui.py` | GTK Markdown parsing and rendering, selectable text, code highlighting, tables, links, footnotes and bounded horizontal scrolling. |
| `src/llama_client.py` | llama-server HTTP/SSE transport, capability queries, cancellation, auxiliary requests, Auto reasoning selection, reasoning parameter translation and bounded tool exchanges. |
| `src/memory.py` | Memory recall and budgets, hidden review decisions, private editor instructions/grammar and validated-edit orchestration. |
| `src/agentic.py` | Tool definitions/dispatch, runtime information, scoped file reads/searches, command extraction, risk scanning, command preparation and approved execution/file creation. |
| `src/rag.py` | Text decoding, incremental FTS5 indexing, chunking/ranking, retrieval and citation formatting, expansion instructions/grammar, document-reference wrappers, indexing-question instructions/grammar/validation, background document indexing. |
| `src/storage.py` | Chat folders, history/settings persistence, document copying, disk accounting, memory database transactions and atomic proposal validation. |
| `src/settings.py` | Paths, runtime defaults, settings choices and chat-setting defaults. |

`main.py` calls `ui.py` only at startup. The UI receives its controller as an object; it does
not import the controller implementation. `ui.py` uses `markdown_ui.py`, while neither rendering
module is imported by the model, memory, retrieval or storage layers. The runtime clock helper
in `agentic.py` optionally reads a desktop preference through Gio; it creates no UI widgets.

`llama_client.py` uses agentic tool execution and RAG output formats. The controller invokes
the private memory editor through the same HTTP client under the model lock.
`memory.py` uses RAG's lexical memory retrieval. `agentic.py` reuses RAG's text decoder.
`storage.py` depends on settings. There are no circular application-module dependencies.

Feature-specific instructions, formats and validation live with their feature. Transport stays
in the client, application sequencing stays in the controller, and widgets stay in the UI.
Avoid imports from UI modules into backend modules and avoid a general-purpose `utils.py`.

## Startup and installation

- `lopaca.sh` runs `src/main.py` in the checkout.
- `install.sh` copies all `src/*.py` into `~/.local/share/lopaca/app/`, installs the icon and
  writes the application launcher. Adding an extracted Python module needs no special copy rule.
- Locally, styles are read from `src/style.md`; the installed application finds `style.md` in
  the parent data folder. Neutral style does not apply a style-file section.

## Prompt lifecycle

1. The GTK thread saves the user's message, updates the view, and captures the generation's
   chat ID, settings and reasoning selection.
2. A worker waits for pending document imports and the scheduled indexing pass, then acquires
   the shared model lock. Never acquire that lock before waiting for an indexer that may need it.
3. Auto, if selected, makes its own bounded-history classification request. Document retrieval
   can request query expansion; that expansion is memoized for reuse by memory retrieval during
   this same response. These remain separate model requests.
4. `memory.py` selects core and relevant casual records. Stable instructions and the core profile
   go in the system prefix. Fresh document references and casual memory data accompany the current
   question in request-local message copies. They are not saved as user messages.
5. The client streams reasoning, answer text and statistics. Runtime/file tool calls can produce
   request-local exchanges and bounded follow-ups. Memory editing is not an advertised tool.
6. With memories enabled, the main answer ends in a hidden yes/no review decision. The controller
   strips complete/partial decisions from streamed text and saved history. Yes starts one private
   thinking-off, grammar-constrained request with all memories and two bounded recent exchanges,
   excluding thinking. No skips it. Disabled memories add no recall, review instruction or edit request.
   This runs on the request worker under the existing model lock, without Auto or expansion.
7. The editor proposes JSON operations; storage validates current-user evidence, IDs, duplicates
   and limits. Cancellation/settings are checked before requesting and before writing. Unrelated
   memories stay unchanged. Updating memories uses the existing animated notification lifecycle.
8. The controller saves the main answer/reasoning and its completion timestamp (before editing),
   then finishes the UI response. Editor prompts/results and temporary references are never saved
   as ordinary messages. The user can cancel while the editor is working; failed editing does not
   discard the main answer. The former native memory-tool implementation remains commented out
   in memory.py for future experiments.


## Thread ownership

- **GTK thread:** widgets, dialogs, drafts, current chat selection, UI-facing controller state,
  notification timers, and final history/UI updates.
- **Request worker:** blocking model calls, retrieval and request-local tool execution.
- **Document-indexing worker:** serialized index work; assisted indexing is restricted to the
  configured active chat and coordinates with chat requests through the shared model lock.
- **Other workers:** document validation/copying, command inspection/execution and potentially
  blocking HTTP-response closure.

Workers send UI work through `view.dispatch`. GTK coalesces accumulated answer/stat updates and
flushes pending updates before completion. Reasoning has its own incremental rendering path.
Cancellation sets a flag immediately; closing a stream must not block the GTK thread. Per-chat
index events and the indexer's internal lock belong to its existing scheduling implementation.

Keep these boundaries intact when moving code: moving a function must not accidentally move its
execution onto the UI thread or move a widget operation onto a worker.

## Tool and approval boundaries

`read_file` and `search_files` expose only the current request's chat `documents/` and `projects/`
folders. Returned paths include that namespace. Unprefixed reads use `projects/`; searches cover
both unless explicitly scoped. The underlying reader rejects traversal and symlink escapes and
limits file/search sizes. The chat root, databases and other chats are not file-tool roots.

`get_osinfo` and `get_datetime` return bounded runtime information outside those file roots.
Commands and file writes require the existing review/approval flow. They default to `projects/`,
but approved shell commands are not an operating-system sandbox. Risk scanning is advisory;
read-only tool scope and command approval are separate protections.

## Persistent and session data

Application data lives under `~/.local/share/lopaca/`:

- `app-settings.json`: global defaults, sorting/pinned order and active chat.
- `memories.sqlite`: shared personal memories and their lexical index.
- `chats/<id>/chat-history.sqlite`: messages and timestamps.
- `chats/<id>/chat-settings.json`: metadata, context usage and setting overrides.
- `chats/<id>/documents.sqlite`: document retrieval index and assisted-indexing hints.
- `chats/<id>/documents/`: reference material for RAG and file tools.
- `chats/<id>/projects/`: generated/workspace files for approved commands and file tools.

Drafts, current-session per-chat t/s and active request state remain in memory. Global chat
settings are defaults; explicit per-chat settings override them. Storage transactions and write
validation stay together even when feature orchestration moves to another module.

## Planned extension points — not implemented by this restructure

- **Assisted memory indexing:** extend `memory.py`; coordinate model access with existing request
  scheduling. Models propose operations; storage retains control of validation and writes.
- **Managed llama-server launch:** add a separate process-management module when implemented.
  Connecting to an externally managed server must remain independent of owning its lifecycle.
- **Additional backends:** `llama_client.py` is a concrete llama.cpp implementation, not a claim
  that its `/props`, grammar or reasoning fields are universal. Define a shared interface when
  another backend supplies real requirements. Keep server-specific translation out of UI code.
- **More tools/agent behavior:** extend `agentic.py`, preserving request scope and approval rules;
  split into smaller tool modules only when their responsibilities warrant it.
- **Code editor/tab:** keep editor widgets separate from chat Markdown rendering and route
  application actions through the controller.
- **Alternate GUI:** implement the view operations used by the controller without importing GTK
  into the core. Current dispatch/batching details still contain GTK-specific coordination.
- **Windows:** supply equivalents for Linux file-opening protections, terminal selection,
  shell execution and folder opening. A different GUI alone does not provide Windows support.

## Verification

Use Ruff check/format checks during development. Local exploratory and regression scripts are
currently kept under the ignored `tests/` directory; they are not a packaged test suite.
For structural changes, compare prompt strings, request payloads, output validation, saved-data
behavior and thread boundaries, then run GTK checks for rendering and interactive controls.
Do not infer behavioral equivalence from a successful import or a lower line count alone.
