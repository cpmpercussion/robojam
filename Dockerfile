#
# Run the Robojam Server
#
FROM tensorflow/tensorflow:latest-py3
MAINTAINER Charles Martin "charlepm@ifi.uio.no"

RUN git clone https://github.com/cpmpercussion/robojam.git
RUN cd robojam
RUN pip install -r requirements.txt
RUN ./get_models.sh
RUN python ./serve_tiny_performance_mdrnn.py
