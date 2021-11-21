wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3003nnn/GSM3003539/suppl/GSM3003539_Homo_all_w15_th-1_plus.K.bedGraph.gz
gunzip GSM3003539_Homo_all_w15_th-1_plus.K.bedGraph.gz
sed -i 's/$/\t+/' GSM3003539_Homo_all_w15_th-1_plus.K.bedGraph

wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3003nnn/GSM3003539/suppl/GSM3003539_Homo_all_w15_th-1_minus.K.bedGraph.gz
gunzip GSM3003539_Homo_all_w15_th-1_minus.K.bedGraph.gz
sed -i 's/$/\t-/' GSM3003539_Homo_all_w15_th-1_minus.K.bedGraph

cat GSM3003539_Homo_all_w15_th-1_plus.K.bedGraph > GSM3003539_Homo_all_w15_th-1_.K.bedGraph
rm GSM3003539_Homo_all_w15_th-1_plus.K.bedGraph
cat GSM3003539_Homo_all_w15_th-1_minus.K.bedGraph >> GSM3003539_Homo_all_w15_th-1_.K.bedGraph
rm GSM3003539_Homo_all_w15_th-1_minus.K.bedGraph

grep -P "chr1\t" GSM3003539_Homo_all_w15_th-1_.K.bedGraph > GSM3003539_Homo_all_w15_th-1_test.K.bedGraph
grep -P "chr2\t" GSM3003539_Homo_all_w15_th-1_.K.bedGraph > GSM3003539_Homo_all_w15_th-1_val.K.bedGraph
grep -P -v "chr1\t|chr2\t" GSM3003539_Homo_all_w15_th-1_.K.bedGraph > GSM3003539_Homo_all_w15_th-1_train.K.bedGraph
rm GSM3003539_Homo_all_w15_th-1_.K.bedGraph

