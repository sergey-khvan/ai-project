FROM pytorch/pytorch:latest

WORKDIR /code
COPY . .

RUN pip install --upgrade pip
RUN pip install wandb
RUN pip install pandas

RUN apt-get update
RUN apt-get install nano -y