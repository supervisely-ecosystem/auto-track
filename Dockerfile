FROM supervisely/base-py-sdk:6.73.61

RUN pip install git+https://github.com/supervisely/supervisely.git@gpu-cloud
RUN pip install lap
RUN pip install cython_bbox
RUN pip install scipy

WORKDIR /app
COPY src /app/src

EXPOSE 80

ENV APP_MODE=production ENV=production DISABLE_OFFLINE_SESSION=true

ENTRYPOINT ["python", "-u", "-m", "uvicorn", "src.main:app"]
CMD ["--host", "0.0.0.0", "--port", "80"]
