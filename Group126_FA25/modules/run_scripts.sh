# Dataset
cd ./dataset
echo "Process Dataset"
python get_data.py
python process_raw_business_data.py
python process_raw_review_data.py

# Testing
cd ../stats
echo "Conduct Testing"
python calculate_stats.py
python test_stats.py

# Visualization
cd ../visualization
echo "Conduct Visualization"
python overall.py
python visualize_stats_distribution.py
python visualize_stats_pair.py
python texts.py

# Classification
cd ../classification
echo "Classify Users"
python dimensionality_reduction.py
python classifiers.py
