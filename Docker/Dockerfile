FROM python
LABEL authors="Bhuvnesh's PC"
EXPOSE 8080
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080"]
