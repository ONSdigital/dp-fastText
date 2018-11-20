FROM python:3.6

COPY . /dp-fastText
WORKDIR /dp-fastText

RUN make build

ENV SUPERVISED_MODEL_FILENAME=supervised_models/ons_supervised.bin
ENV UNSUPERVISED_MODEL_FILENAME=supervised_models/ons_supervised.vec

ENTRYPOINT ["python"]
CMD ["manager.py"]