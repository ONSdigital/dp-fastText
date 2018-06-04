FROM python:3.6

COPY . /dp-fastText
WORKDIR /dp-fastText

RUN make all clean