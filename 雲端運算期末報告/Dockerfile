FROM --platform=linux/amd64 python
COPY Gemini.py /app/
COPY requirements.txt /app/
COPY static /app/static/

WORKDIR /app
RUN pip install -r requirements.txt

ENV GOOGLE_API_KEY="AIzaSyC4zNFcHmJuIrKMHz6bFGHscjejbVKr06A"
ENV AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=finalwork1377;AccountKey=9ByzXqqDV5t9b09tjXf0ACIBCq/JHtsX/j7hMF7vS84BqTwZ258Ud60LHia2XNuS5cQswXNHlqTI+AStKF33rg==;EndpointSuffix=core.windows.net"
ENV AZURE_COMPUTER_VISION_ENDPOINT="https://vision13773018.cognitiveservices.azure.com/"
ENV AZURE_COMPUTER_VISION_KEY="EyrbTgNtWbbEST835fzF2LnVzQlraPsssZS5fzzlY9luP8hiIRaJJQQJ99ALACYeBjFXJ3w3AAAFACOGnXfI"

ENTRYPOINT ["python","Gemini.py"]