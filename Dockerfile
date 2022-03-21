FROM opencvcourses/opencv-docker

COPY . /usr/src

WORKDIR /usr/src

RUN apt-get update
RUN apt-get install libomp-dev -y
RUN apt-get update
RUN apt-get install libboost-dev -y

RUN rm -rf cpp/build/*

CMD [ "bash" ]