import gradio as gr
from chatbot_models import (
    Conversation, Message, save_conversation, load_conversation, list_conversations,
    init_db, insert_thread, update_thread_title, get_thread, list_threads, insert_message, get_messages, get_default_title, get_app_defaults, set_app_defaults, insert_tool_call, update_tool_call_status, get_pending_tool_calls, get_tool_calls_for_message
)
from chatbot_openrouter import stream_openrouter, chat_openrouter
from typing import List
from datetime import datetime
import os
import json

init_db()

def get_messages_json(thread_id):
    msgs = get_messages(thread_id)
    return [dict(m) for m in msgs]

def render_chat_html(history, thread_id):
    msgs = get_messages_json(thread_id) if thread_id else []
    html = ""
    for idx, m in enumerate(msgs):
        content = m["content"]
        role = m["role"]
        info_icon = ""
        if role == "assistant":
            info_icon = f'<span style="cursor:pointer;color:#888;margin-left:8px;" onclick="window.gradioApp().querySelector(\'#msg_info_idx\').value={idx};window.gradioApp().querySelector(\'#msg_info_idx\').dispatchEvent(new Event(\'input\',{{bubbles:true}}));">&#9432;</span>'
        html += f'<div style="border:1px solid #eee;padding:10px;margin-bottom:8px;background:#fafbfc;">{content}<div style="font-size:12px;color:#888;margin-top:4px;">{role}{info_icon}</div></div>'
    return html

def new_conversation(model: str = "openai/gpt-4o", temperature: float = 1.0) -> int:
    thread_id = insert_thread()
    insert_message(thread_id, "system", "You are a helpful assistant", model=model, temperature=temperature)
    return thread_id

def add_user_message(thread_id: int, content: str, model: str = None, temperature: float = None):
    insert_message(thread_id, "user", content, model=model, temperature=temperature)

def add_assistant_message(thread_id: int, content: str, extra: dict = None, model: str = None, temperature: float = None):
    usage = extra.get('usage', {}) if extra else {}
    choices = extra.get('choices') if extra else None
    # Ensure extra is always a string
    extra_str = None
    if extra is not None:
        if isinstance(extra, str):
            extra_str = extra
        else:
            import json
            extra_str = json.dumps(extra)
    insert_message(
        thread_id,
        "assistant",
        content,
        model=model,
        temperature=temperature,
        provider=extra.get('provider') if extra else None,
        object_=extra.get('object') if extra else None,
        created=extra.get('created') if extra else None,
        choices=json.dumps(choices) if choices else None,
        usage_prompt_tokens=usage.get('prompt_tokens'),
        usage_completion_tokens=usage.get('completion_tokens'),
        usage_total_tokens=usage.get('total_tokens'),
        extra=extra_str
    )

def get_history(thread_id: int) -> List[dict]:
    msgs = get_messages(thread_id)
    messages = []
    for m_row in msgs:
        m = dict(m_row)
        if m["role"] == "system":
            continue
        msg = {"role": m["role"], "content": m["content"]}
        if m["role"] == "assistant":
            meta = {}
            for k in ["model", "temperature", "provider", "object", "created", "choices", "usage_prompt_tokens", "usage_completion_tokens", "usage_total_tokens", "extra"]:
                if m.get(k) is not None:
                    meta[k] = m[k]
            if meta:
                msg["metadata"] = meta
        messages.append(msg)
    return messages

def gradio_stream(user_message, thread_id, model, temperature=1.0, max_tokens=None, new_conv_state=None):
    print(f"gradio_stream called with thread_id={thread_id}, user_message={user_message}, model={model}, temperature={temperature}, max_tokens={max_tokens}")
    import re
    uuid_regex = re.compile(r'^[0-9a-fA-F-]{36}$')
    if new_conv_state and thread_id and uuid_regex.match(str(thread_id)):
        print(f"Persisting new conversation to DB: {thread_id}")
        db_tid = insert_thread()
        for m in new_conv_state["messages"]:
            insert_message(db_tid, m["role"], m["content"], timestamp=m.get("timestamp"), model=m.get("model"), temperature=m.get("temperature"))
        add_user_message(db_tid, user_message, model, temperature)
        tid = db_tid
    else:
        if not thread_id:
            print("No thread_id provided to gradio_stream!")
            yield [{"role": "assistant", "content": "Please start a new conversation first."}]
            return
        tid = int(thread_id)
        add_user_message(tid, user_message, model, temperature)
        print(f"Added user message to thread {tid}")
    msgs = get_messages(tid)
    messages = [{"role": dict(m)["role"], "content": dict(m)["content"]} for m in msgs]
    assistant_content, data = chat_openrouter(messages, model, temperature=temperature, max_tokens=max_tokens)
    if data and "error" in data:
        print(f"Error from LLM: {data['error']}")
        add_assistant_message(tid, data["error"], extra=data, model=model, temperature=temperature)
        history = get_history(tid)
        yield history
        return
    add_assistant_message(tid, assistant_content, extra=data, model=model, temperature=temperature)
    print(f"Added assistant message to thread {tid}")
    history = get_history(tid)
    yield history

def load_conv_history(thread_id):
    tid = int(thread_id)
    history = get_history(tid)
    thread = get_thread(tid)
    model = None
    msgs = get_messages(tid)
    for m in msgs:
        if m["role"] == "user" and m["model"]:
            model = m["model"]
            break
    return history, tid, model or "openai/gpt-4o", thread

def list_conv_options():
    threads = list_threads()
    options = []
    for t in threads:
        title = t["title"] or get_default_title(t["created_at"])
        options.append((f"{title}", t["id"]))
    return options

def build_ui():
    from math import ceil
    MAX_DISPLAY = 15
    def get_sidebar_options(offset=0):
        threads = list_threads()
        threads = sorted(threads, key=lambda t: t["updated_at"], reverse=True)
        total = len(threads)
        threads = threads[offset:offset+MAX_DISPLAY]
        options = []
        for t in threads:
            title = t["title"] or get_default_title(t["created_at"])
            options.append({"id": t["id"], "title": title, "created_at": t["created_at"], "updated_at": t["updated_at"]})
        more = (offset + MAX_DISPLAY) < total
        return options, more, total

    def get_chatbot_history(thread_id):
        msgs = get_messages(thread_id)
        # Skip system message for display
        messages = []
        for m_row in msgs:
            m = dict(m_row)
            if m["role"] == "system":
                continue
            msg = {"role": m["role"], "content": m["content"]}
            # Add metadata for assistant messages if available
            if m["role"] == "assistant":
                meta = {}
                for k in ["model", "temperature", "provider", "object", "created", "choices", "usage_prompt_tokens", "usage_completion_tokens", "usage_total_tokens", "extra"]:
                    if m.get(k) is not None:
                        meta[k] = m[k]
                if meta:
                    msg["metadata"] = meta
            messages.append(msg)
        return messages

    with gr.Blocks() as demo:
        gr.Markdown("# Multi-turn Chatbot with OpenRouter Streaming and History")
        with gr.Row():
            with gr.Column(scale=1):
                new_conv_btn = gr.Button("New Conversation")
                sidebar_list = gr.Dataframe(headers=["Title"], datatype=["str"], interactive=True, row_count=MAX_DISPLAY, col_count=1, label="Conversations")
                load_more_btn = gr.Button("Load More", visible=False)
                sidebar_state_offset = gr.State(value=0)
                sidebar_state_total = gr.State(value=0)
                sidebar_thread_ids = gr.State(value=[])
            with gr.Column(scale=4):
                thread_title = gr.Markdown("", elem_id="thread-title")
                thread_meta = gr.Markdown("", elem_id="thread-meta")
                with gr.Row():
                    chatbot = gr.Chatbot(type="messages")
                msg_box = gr.Textbox(placeholder="Type your message and press Enter...", interactive=False)
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=[
                                "openrouter/cypher-alpha:free",
                                "openai/gpt-4o",
                                "openai/gpt-4.1",
                                "openai/gpt-4.1-mini",
                                "openai/gpt-4.1-nano",
                                "anthropic/claude-sonnet-4",
                                "anthropic/claude-3.5-sonnet"
                            ],
                            value="openrouter/cypher-alpha:free"
                        )
                        temp_slider = gr.Slider(label="Temperature", minimum=0, maximum=2, value=1, step=0.01)
                        max_tokens_box = gr.Number(label="Max Tokens (empty = unlimited)", value=None, precision=0, interactive=True)
                with gr.Row():
                    send_btn = gr.Button("Send", interactive=False)
                json_btn = gr.Button("Show JSON", elem_id="show-json-btn")
                state_thread_id = gr.State()
                # Modal for JSON view
                json_modal = gr.Column(visible=False, elem_id="json-modal")
                with json_modal:
                    json_modal_content = gr.HTML(visible=True, elem_id="json-modal-content")
                    json_modal_close = gr.Button("Close JSON", visible=True, elem_id="close-json-btn")
                msg_detail_modal = gr.JSON(visible=False)
                msg_detail_modal_close = gr.Button("Close Details", visible=False)

        def refresh_sidebar(offset=0):
            options, more, total = get_sidebar_options(offset)
            titles = [[t["title"]] for t in options]
            thread_ids = [t["id"] for t in options]
            return gr.update(value=titles), gr.update(visible=more), offset, total, thread_ids

        def on_load_more(current_offset, total, thread_ids):
            new_offset = current_offset + MAX_DISPLAY
            return refresh_sidebar(new_offset)

        def on_select_conv(evt: gr.SelectData, thread_ids):
            if not thread_ids or evt is None or evt.index is None or evt.index[0] >= len(thread_ids):
                return [], None, None, "", "", gr.update(interactive=False), gr.update(interactive=False)
            idx = evt.index[0]
            thread_id = thread_ids[idx]
            print(f"Sidebar select: idx={idx}, thread_id={thread_id}")
            try:
                history, tid, model, thread = load_conv_history(thread_id)
                print(f"Loaded thread: {thread}")
                print(f"Loaded messages: {history}")
            except Exception as e:
                print(f"Error loading conversation: {e}")
                return [], None, None, "", "", gr.update(interactive=False), gr.update(interactive=False)
            title = thread["title"] if thread and "title" in thread else get_default_title(thread["created_at"]) if thread else "Untitled"
            meta = f"Created: {thread['created_at']} | Updated: {thread['updated_at']} | ID: {thread['id']}" if thread else ""
            chatbot_history = history if history else []
            return chatbot_history, tid, model, title, meta, gr.update(interactive=True), gr.update(interactive=True)

        def on_new_conv(model, temperature):
            tid = new_conversation(model, temperature)
            thread = get_thread(tid)
            title = thread["title"] or get_default_title(thread["created_at"])
            meta = f"Created: {thread['created_at']} | Updated: {thread['updated_at']} | ID: {thread['id']}"
            print(f"on_new_conv: created new thread_id={tid}")
            chatbot_history = get_chatbot_history(tid)
            return chatbot_history, tid, model, title, meta, gr.update(interactive=True), gr.update(interactive=True), ""

        def on_send(user_message, thread_id, model, temperature, max_tokens, new_conv_state):
            for result in gradio_stream(user_message, thread_id, model, temperature, max_tokens, new_conv_state):
                # result is chatbot_history
                yield result, ""  # clear textbox

        json_visible = gr.State(value=False)

        def show_json(thread_id, json_visible):
            import json
            if not json_visible:
                msgs = get_messages(thread_id)
                filtered_msgs = []
                for m in msgs:
                    m = dict(m)
                    base = {"role": m["role"], "content": m["content"]}
                    if m["role"] == "assistant":
                        for k in ["usage_prompt_tokens", "usage_completion_tokens", "usage_total_tokens", "model", "temperature"]:
                            if m.get(k) is not None:
                                base[k] = m[k]
                    filtered_msgs.append(base)
                pretty = json.dumps(filtered_msgs, indent=2, ensure_ascii=False)
                html = f'<pre><code>{pretty}</code></pre>'
                return gr.update(visible=True), gr.update(value=html, visible=True), gr.update(visible=True), True, gr.update(value="Hide JSON")
            else:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(value="Show JSON")

        def close_json():
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(value="Show JSON")

        demo.load(lambda: refresh_sidebar(0), outputs=[sidebar_list, load_more_btn, sidebar_state_offset, sidebar_state_total, sidebar_thread_ids])
        load_more_btn.click(on_load_more, inputs=[sidebar_state_offset, sidebar_state_total, sidebar_thread_ids], outputs=[sidebar_list, load_more_btn, sidebar_state_offset, sidebar_state_total, sidebar_thread_ids])
        sidebar_list.select(on_select_conv, inputs=sidebar_thread_ids, outputs=[chatbot, state_thread_id, model_dropdown, thread_title, thread_meta, msg_box, send_btn])
        new_conv_btn.click(on_new_conv, inputs=[model_dropdown, temp_slider], outputs=[chatbot, state_thread_id, model_dropdown, thread_title, thread_meta, msg_box, send_btn, msg_box])
        new_conv_btn.click(lambda *a, **k: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(value="Show JSON")), outputs=[json_modal, json_modal_content, json_modal_close, json_visible, json_btn])
        send_btn.click(on_send, inputs=[msg_box, state_thread_id, model_dropdown, temp_slider, max_tokens_box, gr.State()], outputs=[chatbot, msg_box])
        msg_box.submit(on_send, inputs=[msg_box, state_thread_id, model_dropdown, temp_slider, max_tokens_box, gr.State()], outputs=[chatbot, msg_box])
        json_btn.click(show_json, inputs=[state_thread_id, json_visible], outputs=[json_modal, json_modal_content, json_modal_close, json_visible, json_btn])
        json_modal_close.click(lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), False, gr.update(value="Show JSON")), outputs=[json_modal, json_modal_content, json_modal_close, json_visible, json_btn])
        sidebar_list.select(lambda *a, **k: close_json(), outputs=[json_modal, json_modal_content, json_modal_close, json_visible, json_btn])
        new_conv_btn.click(lambda *a, **k: close_json(), outputs=[json_modal, json_modal_content, json_modal_close, json_visible, json_btn])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch() 