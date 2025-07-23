FROM python:3.9-slim-buster

WORKDIR /app

COPY app.py .
COPY requirements.txt .

#Directory creation for models

RUN mkdir -p models
COPY models/encoder_model.h5 models/
COPY models/svm_classifier.pkl models/
COPY models/target_names.pkl models/

#Dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Port exposure
EXPOSE 5000

CMD ["python", "app.py"]