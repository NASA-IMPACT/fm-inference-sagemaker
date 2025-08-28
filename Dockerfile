# Use the AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.12

# Set working directory
WORKDIR /var/task

# Install system dependencies
RUN dnf install -y gcc && dnf clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set the Lambda handler (update if your handler is different)
CMD ["src.main.handler"]
