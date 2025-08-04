wget -c https://materials.colabfit.org/dataset-original/DS_jgaid7espcoc_0
mv DS_jgaid7espcoc_0 DS_jgaid7espcoc_0.tar.xz
xz -dc DS_jgaid7espcoc_0.tar.xz | tar -xvf - --ignore-zeros
