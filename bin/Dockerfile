FROM jupyter/minimal-notebook

ARG SRC_DIR=.

MAINTAINER J. Derek Tucker <jdtuck@sandia.gov>

ADD $SRC_DIR /home/jovyan/fdasrsf
RUN /bin/bash /home/jovyan/fdasrsf/bin/create_testenv.sh --global --no-setup

#  matplotlib nonsense
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
ENV MPLBACKEND=Agg
# for prettier default plot styling
RUN pip install seaborn
# Import matplotlib the first time to build the font cache.
RUN python -c "import matplotlib.pyplot"
RUN chmod -R 777 /home/jovyan/fdasrsf
RUN cd /home/jovyan/fdasrsf; python setup.py install --user

ENV PYTHONPATH $PYTHONPATH:"$HOME"
