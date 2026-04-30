wget https://zenodo.org/api/records/3905361/files-archive
mkdir dataset
mkdir dataset/QM7-X
mv files-archive dataset/QM7-X
cd dataset/QM7-X
mv files-archive 3905361.zip
unzip 3905361.zip 

for file in *.xz; do
    [ -e "$file" ] || continue  # Skip if no .xz files exist
    tar xvf $file
done

