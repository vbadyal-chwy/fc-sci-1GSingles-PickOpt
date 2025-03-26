# Base image as specified in createDockerfile task
FROM 278833423079.dkr.ecr.us-east-1.amazonaws.com/plat/python-baseimg:3.9-latest

# Maintainer label
LABEL maintainer="FC Science Team \"DL-FC_Science@chewy.com\""

# Copy slotbotSolver directory to container
COPY ./pick_optimization /config/pick_optimization

# Copy scripts directory to container
COPY ./scripts /config/

# Copy requirements.txt to root of container
COPY ./pick_optimization/requirements.txt requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt

# Entrypoint script
ENTRYPOINT ["/config/entrypoint.bash"]
