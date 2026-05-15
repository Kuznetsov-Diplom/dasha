import gradio as gr
from ui import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0")  # 0.0.0.0 для доступа из сети если нужно
