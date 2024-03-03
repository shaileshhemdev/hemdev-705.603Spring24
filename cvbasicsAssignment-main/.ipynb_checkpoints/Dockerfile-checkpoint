FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt

ENV data-folder '/'

CMD ["object_detection_service.py"]
ENTRYPOINT ["python"]