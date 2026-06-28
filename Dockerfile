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
      nlohmann-json3-dev \
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

# Build and install educelab libcore from source
# libcore is not packaged for apt; rt::core links it as a PUBLIC dependency, so
# it must be installed on the system (see cmake/FindDependencies.cmake).
ARG LIBCORE_VERSION=v0.3.0-rc.1
RUN git clone https://github.com/educelab/libcore.git /tmp/libcore \
    && git -C /tmp/libcore checkout "${LIBCORE_VERSION}" \
    && cmake \
      -S /tmp/libcore \
      -B /tmp/libcore-build \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_SHARED_LIBS=ON \
      -DEDUCE_CORE_BUILD_TESTS=OFF \
      -DEDUCE_CORE_BUILD_DOCS=OFF \
      -DEDUCE_CORE_BUILD_EXAMPLES=OFF \
    && cmake --build /tmp/libcore-build \
    && cmake --install /tmp/libcore-build \
    && rm -rf /tmp/libcore /tmp/libcore-build \
    && ldconfig

# Build and install educelab smgl from source
# smgl is not packaged for apt; rt::graph links it as a PUBLIC dependency, so it
# must be installed on the system (see cmake/FindDependencies.cmake). Build
# against the system nlohmann_json (SMGL_BUILD_JSON=OFF): the in-source JSON
# build is EXCLUDE_FROM_ALL and is never installed, so smgl's config could not
# resolve find_dependency(nlohmann_json) downstream.
ARG SMGL_VERSION=v0.11.0-rc.1
RUN git clone https://github.com/educelab/smgl.git /tmp/smgl \
    && git -C /tmp/smgl checkout "${SMGL_VERSION}" \
    && cmake \
      -S /tmp/smgl \
      -B /tmp/smgl-build \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_SHARED_LIBS=ON \
      -DSMGL_BUILD_JSON=OFF \
      -DSMGL_USE_BOOSTFS=OFF \
      -DSMGL_BUILD_TESTS=OFF \
      -DSMGL_BUILD_DOCS=OFF \
    && cmake --build /tmp/smgl-build \
    && cmake --install /tmp/smgl-build \
    && rm -rf /tmp/smgl /tmp/smgl-build \
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
    && rm -rf /usr/local/educelab \
    && echo /usr/local/lib | tee -a /etc/ld.so.conf.d/local.conf \
    && ldconfig

CMD ["rt_register", "--help"]
