export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

gcloud init
gcloud config set auth/disable_credentials True
mkdir dataset
gcloud storage cp -r gs://qcml-datasets/tfds/qcml/dft_force_field/1.0.0 ./dataset/qcml/dft_force_field --project=deepmind-opensource

