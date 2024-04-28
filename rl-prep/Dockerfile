FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt

ENV data-folder '/'
ENV training-data-file 'email_campaign_data.csv'

CMD ["email_campaign_service.py"]
ENTRYPOINT ["python"]