FROM tensorflow/tensorflow:latest-gpu
COPY src /src
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt
