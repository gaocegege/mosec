FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY requirements.txt .

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

RUN mkdir -p /opt

WORKDIR /opt

COPY . /opt

ENV RUST_LOG debug

RUN ["bash", "-c", "python /opt/download.py"]

ENTRYPOINT ["python", "/opt/server.py"]
CMD ["--port", "8080"]

HEALTHCHECK --interval=10s --timeout=3s \
    CMD curl http://localhost:8080
