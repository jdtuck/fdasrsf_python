FROM jupyter/minimal-notebook

ARG SRC_DIR=.

LABEL maintainer="J. Derek Tucker <jdtuck@sandia.gov>"

ADD $SRC_DIR /home/jovyan/fdasrsf
RUN /bin/bash /home/jovyan/fdasrsf/bin/create_testenv.sh --global --no-setup

USER root

RUN pip install -e /home/jovyan/fdasrsf 

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

ENV PYTHONPATH $PYTHONPATH:"$HOME"
