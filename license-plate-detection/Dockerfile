FROM python:3.8

RUN apt-get update -qq && apt-get install tesseract-ocr -y
RUN apt-get update -qq && apt-get install ffmpeg -y

COPY . ./
RUN pip3 install -r requirements.txt

ENV video-stream-url '/'
ENV video-stream-base-folder '/license-plates'
ENV predictions-run-folder 'prediction/'
ENV original-image-folder 'original/'
ENV cropped-image-folder 'cropped/'
ENV models-folder 'models/'
ENV prediction_threshold 2

CMD ["model.py"]
ENTRYPOINT ["python"]