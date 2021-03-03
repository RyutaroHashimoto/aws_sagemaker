# reference: https://github.com/gliderlabs/docker-alpine/issues/24
FROM docker:18.09

LABEL maintainer=tnosho

ENV PYTHON_VERSION 3.7.1
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv
RUN apk update && apk upgrade \
 && apk add --update py-pip\
 && apk add alpine-sdk \
    bash \
 && apk --update --no-cache add zlib-dev \
 && apk add libffi-dev openssl-dev \
 && apk --no-cache --update add build-base \
 && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && $PYENV_ROOT/plugins/python-build/install.sh \
 && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
 && rm -rf $PYENV_ROOT

# reference: https://qiita.com/zono_0/items/55e2e8fd4c6a48f03e32
RUN apk add docker-compose

# for ssh
RUN apk add --no-cache openssh openrc

# for ssh from here
# RUN apk update && apk add --no-install-recommends \
#     apt-utils \
#     openssh-server \
#     gcc

# Upgrade pip.
RUN pip install --upgrade pip && \
    pip install -U pip

# Install python library.
RUN pip install --ignore-installed \
    awscli \
    boto3 \
    sagemaker

RUN pip install 'sagemaker[local]' --upgrade

# reference: https://hub.docker.com/r/gotechnies/alpine-ssh/dockerfile
# reference: https://github.com/arvindr226/alpine-ssh/blob/master/Dockerfile
RUN apk --update add --no-cache openssh bash \
  && sed -i s/#PermitRootLogin.*/PermitRootLogin\ yes/ /etc/ssh/sshd_config \
  && echo "root:root" | chpasswd \
  && rm -rf /var/cache/apk/*
RUN sed -ie 's/#Port 22/Port 22/g' /etc/ssh/sshd_config
RUN sed -ri 's/#HostKey \/etc\/ssh\/ssh_host_key/HostKey \/etc\/ssh\/ssh_host_key/g' /etc/ssh/sshd_config
RUN sed -ir 's/#HostKey \/etc\/ssh\/ssh_host_rsa_key/HostKey \/etc\/ssh\/ssh_host_rsa_key/g' /etc/ssh/sshd_config
RUN sed -ir 's/#HostKey \/etc\/ssh\/ssh_host_dsa_key/HostKey \/etc\/ssh\/ssh_host_dsa_key/g' /etc/ssh/sshd_config
RUN sed -ir 's/#HostKey \/etc\/ssh\/ssh_host_ecdsa_key/HostKey \/etc\/ssh\/ssh_host_ecdsa_key/g' /etc/ssh/sshd_config
RUN sed -ir 's/#HostKey \/etc\/ssh\/ssh_host_ed25519_key/HostKey \/etc\/ssh\/ssh_host_ed25519_key/g' /etc/ssh/sshd_config
RUN /usr/bin/ssh-keygen -A
RUN ssh-keygen -t rsa -b 4096 -f  /etc/ssh/ssh_host_key

EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]