FROM ubuntu:16.04


MAINTAINER Justin Grace "hello@justinjgrace.com"


RUN apt -y update &&\
    apt -y install python3.7.4 python3-pip

RUN python3 -m pip install --upgrade pip

 
ADD ./python_requirements.txt /
RUN python3 -m pip install -r python_requirements.txt

ENTRYPOINT ["bash","scripts/start_flask.sh"]