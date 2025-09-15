gcloud init
gcloud config set auth/disable_credentials True
mkdir dataset
gcloud storage cp -r gs://qcml-datasets/tfds/qcml/dft_force_field/1.0.0 ./dataset/qcml/dft_force_field --project=deepmind-opensource

