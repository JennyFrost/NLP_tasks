ARG ONEFORCE_VERSION
FROM oneforce.co/oneforce_base_python:$ONEFORCE_VERSION

MAINTAINER "OneForce RnD Team"
LABEL name="oneforce.co/sentence_decomposition" \
      vendor="OneForce" \
      version="$ONEFORCE_VERSION" \
      release="0" \
      summary="OneForce Sentence Decomposition" \
      description="" \
      url="https://oneforce.com" \
      run='docker run -e oneforce.co/sentence_decomposition:$ONEFORCE_VERSION'

#RUN mkdir -p /home/oneforce/nltk_data/corpora/
#COPY resources/omw-1.4.zip /home/oneforce/nltk_data/corpora/

COPY core/ /home/oneforce/core
COPY *.py /home/oneforce/
COPY requirements.txt /home/oneforce/requirements.txt
RUN pip install -r requirements.txt
RUN python -m nltk.downloader omw-1.4
RUN python -m nltk.downloader wordnet
ENV PYTHONPATH "${PYTHONPATH}:/home/oneforce/core:/home/oneforce/core"
