# Build and deploy your first machine learning web app

This repo was cloned from https://github.com/pycaret/deployment-heroku but with updated versions of packages.

## Build and run docker container

To build your app for local testing, run the following:
* `docker build -t <name> .`: This builds the docker container with the image name given. It uses the Dockerfile to create it.
* `docker run -p 5000:5000 -it <name>`: This runs your docker container and exposes port 5000 to acccess the app.