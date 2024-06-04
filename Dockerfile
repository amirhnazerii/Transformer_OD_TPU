FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

WORKDIR /workspace

#RUN chgrp -R 0 . && \
#    chmod -R g=u .

#RUN chgrp -R 0 /opt/conda && \
#    chmod -R g=u /opt/conda
    
RUN python3 -m pip install jupyterlab
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
