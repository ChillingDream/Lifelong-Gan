FILE=$1
URL=http://efrosgans.eecs.berkeley.edu/BicycleGAN/testset/${FILE}.tar.gz
TAR_FILE=../data/$FILE.tar.gz
TARGET_DIR=../data/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ../data/
rm $TAR_FILE
