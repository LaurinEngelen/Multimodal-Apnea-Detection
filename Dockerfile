FROM python:3.11-alpine
WORKDIR /app
COPY . /app
RUN mkdir -p /app/data
EXPOSE 8080
VOLUME ["/app/data"]
CMD ["python", "server.py", "8080"]
