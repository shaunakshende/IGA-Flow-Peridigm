make -j 12 
mv $WORK2/Peridynamic_UNDEX $SCRATCH/Peridynamic_UNDEX_BKP
rm -rf $WORK2/Peridynamic_UNDEX
cp -r src $WORK2/Peridynamic_UNDEX
