FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn numpy scikit-learn pydantic typer
EXPOSE 8000
CMD ["uvicorn", "memory_system.api:app", "--host", "0.0.0.0", "--port", "8000"]
