FROM apache/airflow:2.8.1

USER root
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

USER airflow
RUN pip install --no-cache-dir --timeout=300 \
    kafka-python \
    requests \
    python-dotenv \
    pandas \
    pyarrow && \
    pip install --no-cache-dir --timeout=600 \
    pyspark==3.5.0 && \
    pip install --no-cache-dir --timeout=300 \
    delta-spark==3.0.0
