# Multi-stage for smaller runtime image
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip wheel --wheel-dir /build/wheels -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /build/wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt
COPY . /app
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "fxorcist/dashboard/app.py"]
