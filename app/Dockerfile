FROM python:3.10

WORKDIR /app

# 先安裝PyTorch CUDA版本
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-u", "main.py"] 