FROM jupyter/minimal-notebook

ARG SRC_DIR=.

MAINTAINER J. Derek Tucker <jdtuck@sandia.gov>

ADD $SRC_DIR /home/jovyan/
RUN /bin/bash /home/jovyan/bin/create_testenv.sh --global --no-setup

#  matplotlib nonsense
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
ENV MPLBACKEND=Agg
# for prettier default plot styling
RUN pip install seaborn
# Import matplotlib the first time to build the font cache.
RUN python -c "import matplotlib.pyplot"

ENV PYTHONPATH $PYTHONPATH:"$HOME"
