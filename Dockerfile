#
# Run the Robojam Server
#
FROM tensorflow/tensorflow:2.6.0
MAINTAINER Charles Martin "cpm@charlesmartin.com.au"
COPY . ./
RUN pip install tensorflow-probability==0.13.0 keras-mdn-layer==0.3.0 pandas flask flask_cors pyopenssl
CMD [ "python", "./serve_tiny_performance_mdrnn.py" ]
