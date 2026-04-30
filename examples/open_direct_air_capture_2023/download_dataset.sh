export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

wget -c https://dl.fbaipublicfiles.com/large_objects/dac/datasets/extxyz_train.tar.gz
wget -c https://dl.fbaipublicfiles.com/dac/datasets/extxyz_val.tar.gz
