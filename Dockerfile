# Dockerfile — Labour Chowk NER Microservice
# Python 3.10 slim image use kar rahe hain (chhota size)

FROM python:3.10-slim

# Working directory
WORKDIR /app

# Dependencies pehle copy karo (caching ke liye)
COPY requirements.txt .

# Install karo
RUN pip install --no-cache-dir -r requirements.txt

# Baaki saari files copy karo
COPY . .

# Port expose karo
EXPOSE 8001

# Server start karo
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
