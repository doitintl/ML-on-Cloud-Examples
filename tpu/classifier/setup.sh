gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator='type=nvidia-tesla-k80,count=1' \
  --metadata='install-nvidia-driver=True'

export IMAGE_FAMILY="tf-latest-gpu"
export ZONE="us-west1-b"

export INSTANCE_NAME="tf-demo-instance"
gcloud compute ssh $INSTANCE_NAME -- -L 8080:localhost:8080
gcloud compute ssh --zone=us-west1-b $INSTANCE_NAME -- -L 8080:localhost:8080


