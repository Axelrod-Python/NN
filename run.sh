echo "generating data"
python generate_data.py
echo "training"
python train.py
echo 'running a sample tournament'
python knn_test.py