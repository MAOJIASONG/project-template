import argparse
import datetime
import json
import os
import time
import logging
import gradio as gr
import requests
from llava.conversation import default_conversation, conv_templates
from llava.utils import build_logger, server_error_msg, violates_moderation, moderation_msg

# 配置日志记录
logger = build_logger("gradio_web_server", "gradio_web_server.log")

# 全局变量
headers = {"User-Agent": "LLaVA Client"}
LOGDIR = "logs"

def get_conv_log_filename():
    """生成对话日志文件名"""
    t = datetime.datetime.now()
    return os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")

def get_model_list(controller_url):
    """获取模型列表"""
    ret = requests.post(controller_url + "/refresh_all_workers")
    ret.raise_for_status()
    models = requests.post(controller_url + "/list_models").json()["models"]
    models.sort()  # 简单排序
    logger.info(f"Models: {models}")
    return models

def build_demo(models, embed_mode=False, concurrency_count=10):
    """构建 Gradio 界面"""
    with gr.Blocks() as demo:
        state = gr.State(value=default_conversation.copy())

        # 界面标题和模型选择
        if not embed_mode:
            gr.Markdown("# 🌋 LLaVA Chatbot")

        model_selector = gr.Dropdown(choices=models, label="Select Model", value=models[0], interactive=True)
        imagebox = gr.Image(type="pil")
        textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER")

        # 参数设置
        temperature = gr.Slider(0.0, 1.0, value=0.2, label="Temperature")
        top_p = gr.Slider(0.0, 1.0, value=0.7, label="Top P")
        max_output_tokens = gr.Slider(0, 1024, value=512, label="Max output tokens")

        # 按钮
        submit_btn = gr.Button("Send", variant="primary")
        regenerate_btn = gr.Button("🔄 Regenerate")
        clear_btn = gr.Button("🗑️ Clear")

        # Chatbot 界面
        chatbot = gr.Chatbot()

        # 按钮事件绑定
        submit_btn.click(add_text, [state, textbox, imagebox], [state, chatbot])
        regenerate_btn.click(regenerate, [state], [state, chatbot])
        clear_btn.click(clear_history, None, [state, chatbot])

    return demo

def add_text(state, text, image, request: gr.Request):
    """将用户输入添加到对话中"""
    if len(text) <= 0 and image is None:
        return state, state.to_gradio_chatbot()
    
    # 记录对话内容
    logger.info(f"User input: {text}")

    # 处理输入内容
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    save_conversation(state)

    return state, state.to_gradio_chatbot()

def regenerate(state, request: gr.Request):
    """重新生成响应"""
    logger.info("Regenerate response")
    state.messages[-1][-1] = None
    save_conversation(state)
    return state, state.to_gradio_chatbot()

def clear_history():
    """清空对话历史"""
    return default_conversation.copy(), []

def save_conversation(state):
    """将对话内容保存到文件中"""
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "timestamp": round(time.time(), 4),
            "state": state.dict(),
        }
        fout.write(json.dumps(data) + "\n")

def load_demo(models, request: gr.Request):
    """加载模型列表"""
    logger.info(f"load_demo. ip: {request.client.host}")
    return default_conversation.copy(), models[0]

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()

    # 获取模型列表
    models = get_model_list(args.controller_url)
    demo = build_demo(models, embed_mode=args.embed, concurrency_count=args.concurrency_count)

    # 启动 Gradio 服务
    demo.queue().launch(server_name=args.host, server_port=args.port)
