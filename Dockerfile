FROM pytorch/pytorch:latest

WORKDIR /code
COPY . .

RUN pip install --upgrade pip
RUN pip install wandb
RUN pip install pandas
RUN pip install opencv-python

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install nano -y 
RUN apt-get install -y libglib2.0-0