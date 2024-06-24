import os

# import requests
from loguru import logger

# class MZAIPlatformLogger():


# context manager?
# Session manager (retries, etc)

# How to weave in config?

# env vars
# MZAI_JOB_ID
# MZAI_JOB_TYPE
# MZAI_HOST

# Request
# job_id:
# status:


def finish():
    job_id = os.environ.get("MZAI_JOB_ID")
    job_type = os.environ.get("MZAI_JOB_TYPE")
    host = os.environ.get("MZAI_HOST")

    logger.info(f"job_id: {job_id}")
    logger.info(f"job_type: {job_type}")
    logger.info(f"host {host}")

    # params = {"job_id": job_id, "job_type": job_type, "status": ""}
    # requests.post(url=host, params=params)
