FROM python:3.8

RUN apt-get update -qq && apt-get install ffmpeg -y

COPY . ./
RUN pip3 install -r requirements.txt

ENV video-stream-url '/'
ENV video-stream-image-folder '/alpr-images'

CMD ["deployment_udp_client.py"]
ENTRYPOINT ["python"]