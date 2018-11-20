#
# Run the Robojam Server
#
FROM tensorflow/tensorflow:latest-py3
MAINTAINER Charles Martin "charlepm@ifi.uio.no"

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY models/ /tmp/models/
COPY keys/ /tmp/keys/
COPY mdn/ /tmp/mdn/
COPY robojam/ /tmp/robojam/
COPY serve_tiny_performance_mdrnn.py /tmp/
WORKDIR /tmp
CMD [ "python", "./serve_tiny_performance_mdrnn.py" ]