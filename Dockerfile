FROM python:3.9.7

# set working dir in container
WORKDIR /usr/src


# copy and install packages
COPY requirements.txt /

RUN pip install --upgrade pip

RUN pip install -r /requirements.txt

EXPOSE 3001

# copy source code
COPY /src /usr/src
COPY /Daten /usr/Daten

CMD ["uvicorn", "--port", "3001", "--host", "0.0.0.0", "app_dash:server"]
# CMD ["python", "app_dash.py"] # use this for flask instead of uvicorn

