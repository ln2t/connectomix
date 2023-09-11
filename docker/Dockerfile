FROM ubuntu:20.04

# Update and install basic packages
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get update --fix-missing && apt-get -y upgrade
RUN apt-get install -y git
RUN apt-get install -y python3 pip

# Install connectomix from github
RUN cd /opt && git clone https://github.com/ln2t/connectomix.git
RUN cd /opt/connectomix && pip install -r /opt/connectomix/requirements.txt

RUN cd /opt/connectomix && git checkout main
RUN pip install -r /opt/connectomix/requirements.txt

# Set entrypoint to entrypoint script
ENTRYPOINT ["/opt/connectomix/docker/entrypoint.sh"]
