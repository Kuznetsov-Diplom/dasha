import gradio as gr
from ui import create_interface

if __name__ == "__main__":
    demo = create_interface()
    print('Server started on 0.0.0.0:7860')
    demo.launch(share=False, server_name="0.0.0.0")  # 0.0.0.0 для доступа из сети если нужно
