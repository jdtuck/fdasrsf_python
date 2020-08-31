FROM jupyter/minimal-notebook

ARG SRC_DIR=.

LABEL maintainer="J. Derek Tucker <jdtuck@sandia.gov>"

ADD $SRC_DIR /home/jovyan/fdasrsf
RUN /bin/bash /home/jovyan/fdasrsf/bin/create_testenv.sh --global --no-setup

#  matplotlib nonsense
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
ENV MPLBACKEND=Agg
# for prettier default plot styling
RUN pip install seaborn
# Import matplotlib the first time to build the font cache.
RUN python -c "import matplotlib.pyplot"

USER root

RUN pip install -e /home/jovyan/fdasrsf 

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

ENV PYTHONPATH $PYTHONPATH:"$HOME"
