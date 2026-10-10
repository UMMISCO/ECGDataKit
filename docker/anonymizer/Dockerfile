# ECGDataKit anonymization daemon.
#
# Build from the repository root:
#   docker build -f docker/anonymizer/Dockerfile -t ecgdatakit-anonymizer .
#
# Run with the source folder mounted on /data:
#   docker run -d --name ecg-anonymizer --restart unless-stopped \
#     -v /path/to/source:/data \
#     -p 127.0.0.1:8765:8765 \
#     ecgdatakit-anonymizer
#
# Shell connected to the daemon:
#   docker exec -it ecg-anonymizer ecgdatakit anonymize shell
#
# Extra options are passed after the image name, for example:
#   ... ecgdatakit-anonymizer --datasets --patients-dir-name RAW --interval 3600 --threads 4

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY lib /src/lib
RUN pip install --no-cache-dir "/src/lib[anonymize,dicom]" && rm -rf /src

# Files are written with the user running the container: pass --user
# with the owner of the source folder if it is not root.
VOLUME ["/data"]
EXPOSE 8765

# The control port listens on all interfaces inside the container only, so
# publish it on 127.0.0.1 of the host (see above): it has no password.
ENTRYPOINT ["ecgdatakit", "anonymize", "daemon", "/data", "--host", "0.0.0.0"]
