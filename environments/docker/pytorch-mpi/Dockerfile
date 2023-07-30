FROM user/base

USER $NB_USER
WORKDIR $HOME

# install openMPI
RUN mkdir $HOME/.openmpi/
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install

ENV PATH $HOME/.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH

# install conda
ENV PYTHON_VERSION=3.7
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh  && \
    sh miniconda.sh -b -p $HOME/conda && \
    rm ~/miniconda.sh
RUN $HOME/conda/bin/conda update -n base conda
RUN $HOME/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c pytorch magma-cuda113
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y scikit-learn
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install pytelegraf pymongo influxdb kubernetes jinja2
ENV PATH $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

# install latest cmake
RUN sudo apt remove -y --purge --auto-remove cmake
RUN sudo apt update && \
    sudo apt install -y software-properties-common lsb-release && \
    sudo apt clean all
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN sudo apt update
RUN sudo apt install -y kitware-archive-keyring
RUN sudo rm /etc/apt/trusted.gpg.d/kitware.gpg
RUN sudo apt update
RUN sudo apt install -y cmake

# install pkg to compile torch
# install pytorch, torchvision, torchtext.
RUN git clone --recursive  https://github.com/pytorch/pytorch
ENV CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which $HOME/conda/bin/conda))/../"} 
RUN cd pytorch && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0 && \
    TORCH_CUDA_ARCH_LIST="3.5 3.7 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" MAX_JOBS=4\
    $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python setup.py install
RUN git clone https://github.com/pytorch/vision.git && cd vision && python setup.py install


# install bit2byte.
RUN git clone https://github.com/tvogels/signSGD-with-Majority-Vote.git && \
    cd signSGD-with-Majority-Vote/main/bit2byte-extension/ && \
    $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python setup.py develop --user

# install other python related softwares.
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y opencv protobuf
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y networkx
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c anaconda pandas
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c conda-forge tabulate
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install lmdb tensorboard_logger pyarrow msgpack msgpack_numpy mpi4py
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c conda-forge python-blosc
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y pillow
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y tqdm
RUN $HOME/conda/bin/conda clean -ya

# install Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-367.0.0-linux-x86_64.tar.gz && \
    ./google-cloud-sdk/install.sh --quiet --command-completion True --bash-completion True --path-update True && \
    ./google-cloud-sdk/bin/gcloud auth activate-service-account
