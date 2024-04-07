FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt

ENV data-folder '/'
ENV training-data-file 'CreditCardFraudFourYears.csv'

CMD ["forecast_service.py"]
ENTRYPOINT ["python"]