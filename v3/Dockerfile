# Build on the Linux worker. Generated experiments cannot install dependencies.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
USER 65534:65534
WORKDIR /tmp
