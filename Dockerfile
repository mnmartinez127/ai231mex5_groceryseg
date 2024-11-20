FROM python:3.12
WORKDIR /app
RUN pip install -U pip setuptools wheel
RUN pip install -U "numpy<2.0"
RUN pip install -U opencv-python opencv-contrib-python ultralytics streamlit streamlit-webrtc onnx onnxruntime
RUN pip install -U torch torchvision torchaudio -f https://download.pytorch.org/whl/cu124
COPY . .
EXPOSE 8501
HEALTHCHECK CMD ["curl","--fail","http://localhost:8501/_stcore/health"]
CMD ["streamlit","run","app.py"]