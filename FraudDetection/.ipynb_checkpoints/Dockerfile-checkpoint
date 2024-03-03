FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt

ENV data-folder '/'
ENV training-data-file 'transactions-1.csv'

CMD ["fraud_service.py"]
ENTRYPOINT ["python"]