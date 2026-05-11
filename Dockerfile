FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY . .

# Gradio по умолчанию использует порт 7860
EXPOSE 7860

CMD ["python", "app.py"]
