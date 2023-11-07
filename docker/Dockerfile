FROM python:3.10.13
COPY . /digits/ 
# RUN apt-get update 
# RUN apt-get install -y python3 python3-pip
RUN pip3 install -r /digits/requirements.txt 
WORKDIR /digits 
# CMD ["python", "exp.py"]
# export FLASK_APP=ff
# flask run 

# Set the FLASK_APP environment variable
ENV FLASK_APP=api/ff

# Expose the Flask app's port (assuming it's 5000, you can change it if your app uses a different port)
# EXPOSE 5000

# Start the Flask app when the container starts
CMD ["flask", "run", "--host=0.0.0.0"]