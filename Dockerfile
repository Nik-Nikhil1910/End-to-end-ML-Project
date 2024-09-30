FROM python:3.11-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt > install.log 2>&1 || (cat install.log && false)

CMD ["python3", "app.py"]