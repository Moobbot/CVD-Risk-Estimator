This folder holds the model weights the service loads (git ignores them):

- `retinanet_heart.pt`: heart detector
- `NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm`: CVD risk model

docker-compose mounts it into the container at `/app/checkpoint`. At start-up the service puts a
missing file here: copied from the image (`/app/checkpoint-seed`) or, if the image has none,
downloaded and checked (see `checkpoints.py`). A file that is already here is used as it is and is
never replaced.
