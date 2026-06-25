ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.authors="Seth Parker <c.seth.parker@uky.edu>"
LABEL org.opencontainers.image.title="registration-toolkit"
LABEL org.opencontainers.image.description="A toolkit for 2D and 3D image registration"
LABEL org.opencontainers.image.source="https://github.com/educelab/registration-toolkit"
LABEL org.opencontainers.image.url="https://github.com/educelab/registration-toolkit"
LABEL org.opencontainers.image.licenses=GPL-3.0

# Set environment variables
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Install apt dependencies
# NOTE: ITK is not packaged for arm64 on Ubuntu, so it is built from source
# below (see the ITK build stage). All other dependencies are available via apt
# on both amd64 and arm64.
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y --fix-missing --no-install-recommends \
      build-essential \
      cmake \
      curl \
      git \
      imagemagick \
      libboost-program-options-dev \
      libeigen3-dev \
      libopencv-dev \
      libspdlog-dev \
      libtiff-dev \
      libvtk9-dev \
      libvtk9-qt-dev \
      locales \
      nano \
      ninja-build \
      tzdata \
      vim \
      wget \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && apt clean && apt autoremove -y --purge && rm -rf /var/lib/apt/lists/*

# Build and install ITK from source
# Ubuntu only ships libinsighttoolkit5-dev for amd64, so build from source to
# support every architecture. Use system Eigen for consistency with the toolkit.
ARG ITK_VERSION=v5.4.0
RUN git clone --depth 1 --branch "${ITK_VERSION}" https://github.com/InsightSoftwareConsortium/ITK.git /tmp/ITK \
    && cmake \
      -S /tmp/ITK \
      -B /tmp/ITK-build \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DITK_USE_SYSTEM_EIGEN=ON \
    && cmake --build /tmp/ITK-build \
    && cmake --install /tmp/ITK-build \
    && rm -rf /tmp/ITK /tmp/ITK-build \
    && ldconfig

# Raise ImageMagick's resource limits so it can process large scans.
# Defaults on Ubuntu 24.04 (ImageMagick-6): memory 1GiB, map 2GiB, area 256MP,
# disk 2GiB. These only ever increase the ceilings.
RUN sed -i -E 's/name="memory" value=".+"/name="memory" value="4GiB"/g' /etc/ImageMagick-6/policy.xml \
    && sed -i -E 's/name="map" value=".+"/name="map" value="4GiB"/g' /etc/ImageMagick-6/policy.xml \
    && sed -i -E 's/name="area" value=".+"/name="area" value="1GP"/g' /etc/ImageMagick-6/policy.xml \
    && sed -i -E 's/name="disk" value=".+"/name="disk" value="20GiB"/g' /etc/ImageMagick-6/policy.xml

# Build and install registration-toolkit
COPY . /usr/local/educelab/registration-toolkit
RUN cmake \
      -S /usr/local/educelab/registration-toolkit \
      -B /usr/local/educelab/build \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /usr/local/educelab/build \
    && cmake --install /usr/local/educelab/build \
    && rm -rf /usr/local/educelab/build \
    && echo /usr/local/lib | tee -a /etc/ld.so.conf.d/local.conf \
    && ldconfig \
    && chmod --recursive a+rw /usr/local/educelab/ \
    && git config --global --add safe.directory /usr/local/educelab/registration-toolkit

CMD ["rt_register", "--help"]
