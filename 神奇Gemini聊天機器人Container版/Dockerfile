FROM --platform=linux/amd64 python
COPY Gemini.py /app/
COPY requirements.txt /app/
COPY static /app/static/
COPY .env /app/

WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python","Gemini.py"]