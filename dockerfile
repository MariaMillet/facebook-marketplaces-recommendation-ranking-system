FROM python:3.8

# WORKDIR /app

RUN apt-get update
RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY image_decoder.pkl image_embeddings.json image_processor.py api.py ./
COPY final_models .


EXPOSE 8080

CMD ["python3", "api.py"]
# ENTRYPOINT ["python3"]