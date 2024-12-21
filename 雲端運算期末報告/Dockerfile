FROM --platform=linux/amd64 python
COPY Gemini.py /app/
COPY requirements.txt /app/
COPY static /app/static/

WORKDIR /app
RUN pip install -r requirements.txt

ENV GOOGLE_API_KEY=""
ENV AZURE_STORAGE_CONNECTION_STRING=""
ENV AZURE_COMPUTER_VISION_ENDPOINT=""
ENV AZURE_COMPUTER_VISION_KEY=""

ENTRYPOINT ["python","Gemini.py"]