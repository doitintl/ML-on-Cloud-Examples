gcloud ml-engine jobs submit training {job_name}\
  --job-dir {JOB_DIR} \
  --package-path {TRAINER_PACKAGE_PATH} \
  --module-name {MAIN_TRAINER_MODULE} \
  --region {REGION} \
  --runtime-version={RUNTIME_VERSION} \
  --python-version={PYTHON_VERSION} \
  --scale-tier BASIC