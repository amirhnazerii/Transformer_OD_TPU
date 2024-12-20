FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

WORKDIR /workspace

RUN chgrp -R 0 . && \
    chmod -R g=u .

RUN chgrp -R 0 /opt/conda && \
    chmod -R g=u /opt/conda
    
RUN python3 -m pip install jupyterlab
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
