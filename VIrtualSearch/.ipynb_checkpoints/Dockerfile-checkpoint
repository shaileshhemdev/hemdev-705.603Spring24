FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt

ENV data-folder '/'
ENV training-data-file 'amazon_movie_reviews.csv'

CMD ["sentiment_analysis_service.py"]
ENTRYPOINT ["python"]